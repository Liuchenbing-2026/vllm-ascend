# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.patch.platform.patch_tokenizer_cache import _chat_ids_via_cache
from vllm_ascend.tokenizer_cache import IncrementalTokenizerCache


class FakeTokenizer:
    """Tokenizer stub whose composability can be switched off on demand."""

    def __init__(self, added: dict[str, int], *, composable: bool = True):
        self._added = added
        self._composable = composable

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._added)

    def __call__(self, text, add_special_tokens: bool = True, **kwargs):
        ids = [ord(c) % 997 for c in text]
        if not self._composable and text:
            # A sentinel derived from the whole string - the failure mode a
            # prefix-space pre-tokenizer produces.
            ids = [len(text) % 97] + ids
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


_DEFAULT_ADDED = {"<s>": 0, "</s>": 1}


def _cache(composable=True, added=_DEFAULT_ADDED):
    # Note: `added={}` must stay distinguishable from "not given".
    tokenizer = FakeTokenizer(added, composable=composable)
    return tokenizer, IncrementalTokenizerCache(tokenizer, capacity_gb=0.01)


def test_encode_is_bit_identical_to_the_tokenizer():
    tokenizer, cache = _cache()
    assert cache.enabled
    for text in ("", "plain text", "a<s>b</s>c", "<s><s></s>", "trailing   <s>   leading", "x" * 5000):
        expected = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Cold then warm: both must match.
        assert cache.encode(text) == expected
        assert cache.encode(text) == expected


def test_self_check_disables_when_the_identity_breaks():
    _, cache = _cache(composable=False)
    assert not cache.enabled
    assert not cache.is_eligible(add_special_tokens=False)


def test_disabled_without_added_tokens():
    _, cache = _cache(added={})
    assert not cache.enabled


def test_add_special_tokens_gating():
    # The stub prepends id 1 when add_special_tokens=True, so it is not a
    # no-op and those requests must be refused.
    _, cache = _cache()
    assert cache.is_eligible(add_special_tokens=False)
    assert not cache.is_eligible(add_special_tokens=True)


def test_eviction_does_not_break_correctness():
    tokenizer = FakeTokenizer({"<s>": 0, "</s>": 1})
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=1e-9)
    if not cache.enabled:
        return
    text = "a" * 400 + "<s>" + "b" * 400
    assert cache.encode(text) == tokenizer(text, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------ chat dispatch


class _FakeChatCache:
    def __init__(self, armed: bool, arm_result: bool = True):
        self.chat_path_enabled = armed
        self.chat_arm_attempted = armed
        self._arm_result = arm_result
        self.seen: list[str] = []

    def arm_chat_path(self, probes):
        self.chat_path_enabled = self._arm_result
        return self._arm_result

    def encode(self, text):
        self.seen.append(text)
        return [7, 7, 7]


def _render_recorder(calls):
    def render(**kw):
        calls.append(kw)
        return "TEXT" if kw.get("tokenize") is False else [1, 2]

    return render


def test_chat_path_used_once_armed():
    cache = _FakeChatCache(armed=True)
    calls: list[dict] = []
    out = _chat_ids_via_cache(cache, _render_recorder(calls), {"tokenize": True})
    assert out == [7, 7, 7]
    assert cache.seen == ["TEXT"]
    assert calls == [{"tokenize": False}]


def test_chat_path_arms_on_first_use():
    cache = _FakeChatCache(armed=False, arm_result=True)
    cache.chat_arm_attempted = False
    calls: list[dict] = []
    out = _chat_ids_via_cache(cache, _render_recorder(calls), {"tokenize": True})
    assert out == [7, 7, 7]
    # Probe renders both ways, then the real render.
    assert [c["tokenize"] for c in calls] == [True, False, False]


def test_chat_path_skipped_when_arming_fails():
    cache = _FakeChatCache(armed=False, arm_result=False)
    cache.chat_arm_attempted = False
    calls: list[dict] = []
    assert _chat_ids_via_cache(cache, _render_recorder(calls), {"tokenize": True}) is None
    assert cache.seen == []


def test_chat_path_skipped_when_caller_wants_text():
    cache = _FakeChatCache(armed=True)
    calls: list[dict] = []
    assert _chat_ids_via_cache(cache, _render_recorder(calls), {"tokenize": False}) is None
    assert calls == []


def test_chat_path_skipped_without_cache():
    calls: list[dict] = []
    assert _chat_ids_via_cache(None, _render_recorder(calls), {"tokenize": True}) is None
