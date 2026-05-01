from opencas.embeddings import local_gemma


def test_local_gemma_honors_opencas_torch_threads(monkeypatch):
    calls = []

    monkeypatch.setenv("OPENCAS_TORCH_THREADS", "23")
    monkeypatch.setattr(local_gemma.torch, "set_num_threads", lambda value: calls.append(value))
    monkeypatch.setattr(local_gemma.torch, "get_num_threads", lambda: calls[-1] if calls else 18)

    assert local_gemma._configure_torch_threads() == 23
    assert calls == [23]


def test_local_gemma_ignores_invalid_thread_count(monkeypatch):
    calls = []

    monkeypatch.setenv("OPENCAS_TORCH_THREADS", "invalid")
    monkeypatch.setattr(local_gemma.torch, "set_num_threads", lambda value: calls.append(value))
    monkeypatch.setattr(local_gemma.torch, "get_num_threads", lambda: 18)

    assert local_gemma._configure_torch_threads() == 18
    assert calls == []
