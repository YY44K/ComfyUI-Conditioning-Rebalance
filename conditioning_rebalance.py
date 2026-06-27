import torch


def _scale_cond_tensor(t: torch.Tensor, multiplier, per_layer_weights=None):
    """Scale a conditioning tensor, optionally with per-layer rebalancing.

    If per_layer_weights is given, rebalance the 12 flattened Qwen taps while
    preserving the original overall RMS magnitude. The global multiplier is
    applied afterward.
    """
    if per_layer_weights is None:
        return t * multiplier

    flat = t.shape[-1]
    n_layers = len(per_layer_weights)

    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype

        # Work in float32 for stable RMS math.
        x = t.float()

        # Save original overall conditioning magnitude.
        orig_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        # Reshape flattened layer stack:
        # (B, seq, 12 * D) -> (B, seq, 12, D)
        x = x.reshape(*x.shape[:-1], n_layers, layer_dim)

        gains = torch.tensor(per_layer_weights, dtype=x.dtype, device=x.device)
        x = x * gains.view(*([1] * (x.dim() - 2)), n_layers, 1)

        # Flatten back:
        # (B, seq, 12, D) -> (B, seq, 12 * D)
        x = x.reshape(*x.shape[:-2], flat)

        # Restore original overall magnitude after changing relative layer balance.
        new_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x = x * (orig_rms / new_rms.clamp_min(1e-6))

        # Apply global strength last.
        return x.to(orig_dtype) * multiplier

    return t * multiplier


def _parse_per_layer(s: str):
    """Parse a comma-separated list of floats. Returns None if empty/invalid."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals


def scale_conditioning(structure, multiplier, per_layer_weights=None):
    """leaving masks / pooled output intact."""
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, multiplier, per_layer_weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, multiplier, per_layer_weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, multiplier, per_layer_weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, multiplier, per_layer_weights)
                for k, v in structure.items()}
    return structure


class ConditioningKrea2Rebalance:

    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "multiplier": ("FLOAT", {"default": 4.0, "min": -1000000000.0, "max": 1000000000.0, "step": 0.01}),
            "per_layer_weights": ("STRING", {"default": cls.DEFAULT_WEIGHTS, "multiline": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning, multiplier, per_layer_weights=None):
        plw = _parse_per_layer(per_layer_weights) if per_layer_weights else None
        c = scale_conditioning(conditioning, multiplier, per_layer_weights=plw)
        return (c,)
