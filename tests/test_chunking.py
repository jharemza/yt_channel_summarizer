import pytest

from summarize import chunk_text_by_tokens


class DummyEncoder:
    def encode(self, text: str):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_chunk_text_by_tokens_basic():
    enc = DummyEncoder()
    text = "one two three four five"
    chunks = chunk_text_by_tokens(text, chunk_size=2, overlap=1, enc=enc)
    assert chunks == [
        "one two",
        "two three",
        "three four",
        "four five",
    ]


def test_chunk_text_by_tokens_single_chunk():
    enc = DummyEncoder()
    text = "one two"
    chunks = chunk_text_by_tokens(text, chunk_size=10, overlap=0, enc=enc)
    assert chunks == ["one two"]
