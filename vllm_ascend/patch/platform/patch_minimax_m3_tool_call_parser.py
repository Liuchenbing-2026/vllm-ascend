#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MiniMax-M3 tool-call parser registration.
# ----------------------------------------------------------------------------
# There is no upstream MiniMax-M3 tool parser. M3 emits the SAME inner tool-call
# XML as MiniMax-M2:
#
#     <tool_call>
#       <invoke name="my_tool">
#         <parameter name="arg1">value1</parameter>
#         ...
#       </invoke>
#       ...
#     </tool_call>
#
# but the WRAPPER tokens differ:
#   * M2 uses  <minimax:tool_call> / </minimax:tool_call>
#   * M3 uses  <tool_call> / </tool_call>   (added_tokens 200052 / 200053,
#     confirmed from the M3 chat_template.jinja: toolcall_begin/end tokens).
#
# Strategy: subclass the existing MinimaxM2ToolParser (which already implements
# the full complete + streaming parsing, AND has the vllm-ascend M2 streaming
# backport monkeypatch applied to it at import) and only swap the start/end
# sentinel tokens + their regexes. Register the subclass under "minimax_m3".
#
# Serve with:  --tool-call-parser minimax_m3  --enable-auto-tool-choice

from __future__ import annotations

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParserManager
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser

# Importing this module ensures the vllm-ascend M2 streaming backport
# monkeypatch is applied to MinimaxM2ToolParser BEFORE we subclass it, so the
# M3 subclass inherits the patched streaming behaviour.
try:  # pragma: no cover - import side-effect ordering safety
    import vllm_ascend.patch.platform.patch_minimax_m2_tool_call_parser  # noqa: F401
except Exception:  # noqa: BLE001
    # If the M2 patch module is absent we still register the parser using
    # whatever MinimaxM2ToolParser implementation is available.
    pass

logger = init_logger(__name__)


@ToolParserManager.register_module("minimax_m3")
class MinimaxM3ToolParser(MinimaxM2ToolParser):
    """MiniMax-M3 tool parser.

    Identical to MiniMax-M2 except the tool-call wrapper tokens are
    <tool_call> / </tool_call> instead of <minimax:tool_call> /
    </minimax:tool_call>.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # ---- Override the sentinel tokens to the M3 variant ----------------
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        # Recompile the wrapper regex against the new tokens. The invoke /
        # parameter inner regexes are unchanged (same XML as M2) and remain as
        # set by the base class.
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )

        # ---- Re-resolve the special-token ids for M3 ----------------------
        # The base __init__ already raised if the M2 tokens were missing; but
        # the M3 tokenizer has <tool_call>/</tool_call> (added_tokens
        # 200052/200053), not the M2 tokens. Re-look them up here.
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        # TODO(verify): M3 prepends an internal namespace separator token
        # ("]<]minimax[>[", id 200058) before <tool_call> in the rendered
        # stream. The base parser keys off the <tool_call> SUBSTRING (via
        # find / `in`), so the ns prefix should not break detection. Confirm
        # against a real M3 tool-calling sample once the model serves; if the
        # ns token is fused with <tool_call> into a single vocab id, streaming
        # detection by token-id may need that fused id instead.
        if (
            self.tool_call_start_token_id is None
            or self.tool_call_end_token_id is None
        ):
            logger.warning(
                "MiniMax-M3 tool parser could not find <tool_call>/</tool_call> "
                "token ids in the tokenizer vocab; streaming tool-call "
                "detection by token-id will be disabled (substring detection "
                "still works). Verify the M3 tokenizer special tokens."
            )

        logger.debug(
            "vLLM Ascend successfully registered MiniMax-M3 tool parser."
        )


# TODO(verify): M3 also produces reasoning/thinking content. M2 uses a separate
# reasoning parser. If M3 needs one, register a "minimax_m3" reasoning parser
# analogously (mirror the M2 reasoning parser). Phase-1 deliverable focuses on
# the tool-call parser only.
