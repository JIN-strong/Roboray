import json

import numpy as np
import pytest

from roboray.models.semantic_models import OpenAIVlm, VlmError


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"choices": [_FakeChoice(self._content)]})()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("VLM_API_KEY", "test-key")
    yield


def test_label_writes_parsed_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    client = _FakeClient(json.dumps(["reach", "grasp"]))
    model = OpenAIVlm(client_factory=lambda **kw: client)
    out = tmp_path / "sem.npz"
    model.label("segments=2", [], out, n_segments=2)
    z = np.load(out, allow_pickle=True)
    assert [str(x) for x in z["labels"]] == ["reach", "grasp"]
    assert client.chat.completions.calls[0]["model"] == "qwen-vl"


def test_label_pads_and_truncates_to_segment_count(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient(json.dumps(["a"])))
    out = tmp_path / "sem.npz"
    model.label("p", [], out, n_segments=3)
    assert len([str(x) for x in np.load(out, allow_pickle=True)["labels"]]) == 3


def test_missing_base_url_raises_naming_the_config_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBORAY_VLM_BASE_URL", raising=False)
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("[]"))
    with pytest.raises(VlmError) as exc:
        model.label("p", [], tmp_path / "s.npz", n_segments=1)
    assert "vlm_base_url" in str(exc.value) or "ROBORAY_VLM_BASE_URL" in str(exc.value)


def test_missing_api_key_raises_naming_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("[]"))
    with pytest.raises(VlmError) as exc:
        model.label("p", [], tmp_path / "s.npz", n_segments=1)
    assert "VLM_API_KEY" in str(exc.value)


def test_non_json_reply_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    model = OpenAIVlm(client_factory=lambda **kw: _FakeClient("this is not JSON"))
    out = tmp_path / "s.npz"
    model.label("p", [], out, n_segments=2)  # must not raise
    labels = [str(x) for x in np.load(out, allow_pickle=True)["labels"]]
    assert len(labels) == 2


_URI = "data:image/jpeg;base64," + "QUJD"  # "ABC"


def test_label_sends_keyframes_as_multimodal_content(tmp_path, monkeypatch):
    """spec §9: the AiFunction input must be keyframes + trajectory summary, not
    plain text."""
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    client = _FakeClient(json.dumps(["reach"]))
    model = OpenAIVlm(client_factory=lambda **kw: client)
    images = [_URI + str(i) for i in range(3)]
    model.label("trajectory summary", images, tmp_path / "s.npz", n_segments=1)
    content = client.chat.completions.calls[0]["messages"][0]["content"]
    assert isinstance(content, list)
    texts = [p for p in content if p["type"] == "text"]
    imgs = [p for p in content if p["type"] == "image_url"]
    assert len(texts) == 1 and texts[0]["text"] == "trajectory summary"
    assert len(imgs) == len(images) == 3
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in imgs)
    assert [p["image_url"]["url"] for p in imgs] == images


def test_label_without_images_stays_text_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORAY_VLM_BASE_URL", "https://vlm.example/v1")
    client = _FakeClient(json.dumps(["reach"]))
    model = OpenAIVlm(client_factory=lambda **kw: client)
    model.label("trajectory summary", [], tmp_path / "s.npz", n_segments=1)
    assert client.chat.completions.calls[0]["messages"][0]["content"] == "trajectory summary"
