"""Stage 2: MobileVLM V2 (+ optional LoRA adapter) explanation generator
(thesis Section 3.5, Appendices B.3 and E).

Follows the official MobileVLM inference recipe (scripts/inference.py in
https://github.com/Meituan-AutoML/MobileVLM, Apache-2.0): the prompt must contain the
`<image>` token and the image tensor is passed as `images=` -- passing `pixel_values` with a
plain-text prompt (as the original demo script did) does not condition the model on the image.
Put a clone of that repository on PYTHONPATH.
"""
from __future__ import annotations

import cv2
import torch
from PIL import Image

from .explain import PROMPT_JSON, parse_explanation


class MobileVLMExplainer:
    def __init__(self, base_dir: str, adapter_dir: str | None = None, merge_adapter: bool = True,
                 conv_mode: str = "v1", device: str = "cuda"):
        from mobilevlm.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from mobilevlm.conversation import conv_templates
        from mobilevlm.model.mobilevlm import load_pretrained_model
        from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token
        disable_torch_init()
        # NB: the official loader has no `vision_tower=` argument; the CLIP tower named in the
        # model config is fetched automatically.
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            base_dir, device=device)
        if adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            if merge_adapter:
                self.model = self.model.merge_and_unload()
        self.model.eval()
        self._img_tok, self._img_idx = DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        self._conv_templates, self._conv_mode = conv_templates, conv_mode
        self._process_images, self._tok_image = process_images, tokenizer_image_token

    @torch.inference_mode()
    def explain(self, frame_bgr, prompt: str = PROMPT_JSON, max_new_tokens: int = 200,
                temperature: float = 0.2) -> tuple[dict, str]:
        """Returns (parsed explanation dict, raw generated text)."""
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        images = self._process_images([img], self.image_processor, self.model.config)
        images = images.to(self.model.device, dtype=torch.float16)
        conv = self._conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], self._img_tok + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = self._tok_image(conv.get_prompt(), self.tokenizer, self._img_idx,
                                    return_tensors="pt").unsqueeze(0).to(self.model.device)
        out = self.model.generate(input_ids, images=images, do_sample=temperature > 0,
                                  temperature=temperature if temperature > 0 else None,
                                  max_new_tokens=max_new_tokens, use_cache=True)
        raw = self.tokenizer.batch_decode(out[:, input_ids.shape[1]:],
                                          skip_special_tokens=True)[0].strip()
        return parse_explanation(raw), raw
