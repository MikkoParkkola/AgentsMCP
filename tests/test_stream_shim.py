from agentsmcp.stream import generate_stream_from_text


def test_generate_stream_from_text_chunks_and_final():
    text = "Hello world" * 5
    chunks = list(generate_stream_from_text(text, step=5))
    assert len(chunks) >= 3
    assert any(c.is_final for c in chunks)
    assembled = "".join(c.text for c in chunks)
    assert assembled == text

