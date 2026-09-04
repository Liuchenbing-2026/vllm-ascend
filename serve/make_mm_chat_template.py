"""Derive a multimodal-capable chat template from the one shipped in the checkpoint.

The released GLM-5.3-Flash chat_template.jinja is the TEXT-ONLY template: its
visible_text() macro replaces any image/video content part with
  "<reminder>You are unable to process this image ...</reminder>"
so the rendered prompt contains no <|image|> placeholder, and vLLM's
Glm4vMultiModalProcessor._get_prompt_updates (which does
PromptReplacement(target=hf_processor.image_token, ...)) has nothing to replace ->
AssertionError: Failed to apply prompt replacement for mm_items['image'][0].

This rewrites that one branch to emit the placeholders the processor expects:
  image -> <|begin_of_image|><|image|><|end_of_image|>
  video -> <|begin_of_video|><|video|><|end_of_video|>
Everything else in the template is byte-identical to the shipped one.
"""
import io, sys

import os
src = os.environ.get("GLM53_MODEL_DIR", "/data02/GLM-5.3-Flash-w8a8-b0829") + "/chat_template.jinja"
dst = sys.argv[1] if len(sys.argv) > 1 else "/data02/glm53_quant/chat_template_mm.jinja"

s = io.open(src, encoding="utf-8").read()

old = """            {%- elif item is mapping and item.type in ['image', 'image_url', 'video', 'video_url', 'audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}"""

new = """            {%- elif item is mapping and item.type in ['image', 'image_url'] -%}
                {{- '<|begin_of_image|><|image|><|end_of_image|>' }}
            {%- elif item is mapping and item.type in ['video', 'video_url'] -%}
                {{- '<|begin_of_video|><|video|><|end_of_video|>' }}
            {%- elif item is mapping and item.type in ['audio', 'audio_url', 'input_audio'] -%}
                {%- set media_type = item.type | replace('_url', '') | replace('input_', '') -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}"""

n = s.count(old)
print("occurrences of the text-only media branch:", n)
assert n >= 1, "anchor not found -- template changed"
s = s.replace(old, new)
io.open(dst, "w", encoding="utf-8").write(s)
print("wrote", dst)
print("image placeholder present:", "<|image|>" in s)
