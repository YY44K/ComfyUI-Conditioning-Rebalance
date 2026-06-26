import math

import torch

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _split_bands(t, n_bands=12):
    flat = t.shape[-1]
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    if ref_dir.shape[0] == 1 and target_batch != 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    if ref_dir.shape[0] != target_batch:
        ref_dir = ref_dir.mean(dim=0, keepdim=True).expand(target_batch, *ref_dir.shape[1:])
    return ref_dir


def _parse_floats(s):
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

# ignored
SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Target longest-side resolution per tier. Each image is scaled to it selected resolution independently
RESOLUTIONS = {"low": 256, "normal": 512, "high": 1024, "max": 1280}


def _scale_to_resolution(samples, target):

    n, c, h, w = samples.shape
    if h == target and w == target:
        return samples
    scale = target / max(h, w)
    nh = max(1, round(h * scale))
    nw = max(1, round(w * scale))
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def compile_edit(clip, prompt, images_with_size=None):
    """Encode an edit prompt with optional reference images."""
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

    images_vl = []
    image_prompt = ""

    if images_with_size:
        for i, (image, tier) in enumerate(images_with_size):
            if image is None:
                continue
            target = RESOLUTIONS.get(tier, 256)
            samples = image.movedim(-1, 1)  # NHWC -> NCHW
            scaled = _scale_to_resolution(samples, target)
            images_vl.append(scaled.movedim(1, -1))  # back to NHWC for clip.tokenize
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    full_prompt = image_prompt + prompt if image_prompt else prompt

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl if images_vl else None,
        llama_template=SYS_TEMPLATE,
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    return conditioning


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=12):
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign=+1)
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def dissim_guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(dissim_guidance_conditioning(item, ref_item, strength, per_band_strengths))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths)
        return structure
    return structure


def guidance(conditioning, reference, strength):
    return dissim_guidance_conditioning(conditioning, reference, strength, per_band_strengths=None)


class Krea2EditRebalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "refocus_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "guidance_strength": ("FLOAT", {"default": 0.500, "min": 0.0, "max": 2.0, "step": 0.01}),
            "enable_split": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    @staticmethod
    def _process_cond(cond, refocus_strength=1.00, guidance_strength=0.500):
        cond_ref = refocus(
            cond, refocus_strength, "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        cond_main = refocus(
            cond, refocus_strength, "0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,9.0,1.0,1.0,1.0",
        )
        return guidance(cond_main, cond_ref, guidance_strength)

    def main(self, text, clip, refocus_strength=0.80, guidance_strength=0.500, enable_split=True,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "(Subject:2) {}".format(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        if enable_split:
            cond_text = compile_edit(clip, prompt, None)
            cond_text = self._process_cond(cond_text, refocus_strength, guidance_strength)
            cond_text = node_helpers.conditioning_set_values(
                 cond_text, {"start_percent": 0.000, "end_percent": 0.175},
            )

            if has_image:
                cond_image = compile_edit(clip, prompt, images_with_size)
                cond_image = self._process_cond(cond_image, refocus_strength, guidance_strength)
                cond_image = node_helpers.conditioning_set_values(
                     cond_image, {"start_percent": 0.175, "end_percent": 1.000},
                )
                final = cond_image + cond_text
            else:
                final = cond_text
        else:
            if has_image:
                final = compile_edit(clip, prompt, images_with_size)
            else:
                final = compile_edit(clip, prompt, None)
            final = self._process_cond(final, refocus_strength, guidance_strength)
            final = node_helpers.conditioning_set_values(
                final, {"start_percent": 0.000, "end_percent": 1.000},
            )

        return (final,)


class Krea2EditRebalanceC:
    """Advanced variant of Krea2EditRebalance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "positive_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "negative_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "positive_layers": ("STRING", {"default": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0", "multiline": False}),
            "negative_layers": ("STRING", {"default": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0", "multiline": False}),
            "guidance_strength": ("FLOAT", {"default": 0.500, "min": 0.0, "max": 2.0, "step": 0.01}),
            "enable_step": ("FLOAT", {"default": 0.000, "min": 0.000, "max": 1.000, "step": 0.001}),
            "enable_split": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "image1": ("IMAGE",),
            "image1_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image2": ("IMAGE",),
            "image2_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image3": ("IMAGE",),
            "image3_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
            "image4": ("IMAGE",),
            "image4_tokens": (["low", "normal", "high", "max"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    @staticmethod
    def _process_cond(cond, positive_strength, negative_strength,
                      positive_layers, negative_layers, guidance_strength):
        cond_negative = refocus(cond, negative_strength, negative_layers)
        cond_positive = refocus(cond, positive_strength, positive_layers)
        return guidance(cond_positive, cond_negative, guidance_strength)

    def main(self, text, clip, positive_strength=1.00, negative_strength=1.00,
             positive_layers="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
             negative_layers="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
             guidance_strength=0.500, enable_step=0.000, enable_split=True,
             image1=None, image1_tokens="normal",
             image2=None, image2_tokens="normal",
             image3=None, image3_tokens="normal",
             image4=None, image4_tokens="normal"):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Edit requires ComfyUI (comfy.utils, node_helpers).")

        prompt = "(Subject:2) {}".format(text)

        images_with_size = [
            (image1, image1_tokens),
            (image2, image2_tokens),
            (image3, image3_tokens),
            (image4, image4_tokens),
        ]
        has_image = any(img is not None for img, _ in images_with_size)

        step = float(enable_step)

        if enable_split:
            cond_text = compile_edit(clip, prompt, None)
            cond_text = self._process_cond(
                cond_text, positive_strength, negative_strength,
                positive_layers, negative_layers, guidance_strength,
            )
            cond_text = node_helpers.conditioning_set_values(
                cond_text, {"start_percent": 0.000, "end_percent": step},
            )

            if has_image:
                cond_image = compile_edit(clip, prompt, images_with_size)
                cond_image = self._process_cond(
                    cond_image, positive_strength, negative_strength,
                    positive_layers, negative_layers, guidance_strength,
                )
                cond_image = node_helpers.conditioning_set_values(
                    cond_image, {"start_percent": step, "end_percent": 1.000},
                )
                final = cond_image + cond_text
            else:
                final = cond_text
        else:
            if has_image:
                final = compile_edit(clip, prompt, images_with_size)
            else:
                final = compile_edit(clip, prompt, None)
            final = self._process_cond(
                final, positive_strength, negative_strength,
                positive_layers, negative_layers, guidance_strength,
            )
            final = node_helpers.conditioning_set_values(
                final, {"start_percent": 0.000, "end_percent": 1.000},
            )

        return (final,)


NODE_CLASS_MAPPINGS = {
    "Krea2EditRebalance": Krea2EditRebalance,
    "Krea2EditRebalanceC": Krea2EditRebalanceC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2EditRebalance": "Krea 2 Image Edit Rebalance",
    "Krea2EditRebalanceC": "Krea 2 Image Edit Rebalance C.",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
