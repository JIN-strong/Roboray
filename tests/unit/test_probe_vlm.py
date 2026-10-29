"""probe_vlm: inject a fake client to cover all 4 tag branches + the dummy key path."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import probe_vlm


class _AuthErr(Exception):
    status_code = 401


def _client(models_ids=("m1",), text_fail=None, mm_fail=None):
    """Build a fake OpenAI client. text_fail/mm_fail are exceptions to raise, or None."""
    def create(model, messages, **kw):
        is_mm = isinstance(messages[0]["content"], list)
        fail = mm_fail if is_mm else text_fail
        if fail is not None:
            raise fail
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="pong"))])

    completions = types.SimpleNamespace(create=create)
    models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(
        data=[types.SimpleNamespace(id=i) for i in models_ids]))
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(models=models, chat=chat)


def _factory(client):
    return lambda **kw: client


def test_all_ok_returns_reachable_multimodal():
    tag, details = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(_client()))
    assert tag == "reachable_multimodal"
    assert any("models" in d for d in details)


def test_multimodal_rejected_returns_text_only():
    c = _client(mm_fail=ValueError("image not supported"))
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "reachable_text_only"


def test_auth_failed_on_text():
    c = _client(text_fail=_AuthErr("bad key"))
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "auth_failed"


def test_unreachable_on_models_and_classify_fallback():
    # non-auth exceptions (no status_code, no "auth" in the name) must fall back to unreachable
    assert probe_vlm.classify_exc(ConnectionError("refused")) == "unreachable"
    assert probe_vlm.classify_exc(ValueError("boom")) == "unreachable"
    c = _client(text_fail=ConnectionError("refused"))
    models = types.SimpleNamespace(list=lambda: (_ for _ in ()).throw(ConnectionError("refused")))
    c = types.SimpleNamespace(models=models, chat=c.chat)
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=_factory(c))
    assert tag == "unreachable"


def test_probe_never_raises_on_arbitrary_failure():
    # any exception becomes a tag; never re-raised
    def bad_factory(**kw):
        raise RuntimeError("anything")
    tag, _ = probe_vlm.probe("http://x", "m", "k", client_factory=bad_factory)
    assert tag == "unreachable"


def test_resolve_key_dummy_and_env(monkeypatch):
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    key, lines = probe_vlm.resolve_key("VLM_API_KEY")
    assert key == "EMPTY" and any("dummy" in l for l in lines)
    monkeypatch.setenv("VLM_API_KEY", "sk-real")
    key, lines = probe_vlm.resolve_key("VLM_API_KEY")
    assert key == "sk-real"
