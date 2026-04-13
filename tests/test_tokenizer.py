"""Tests for the byte-level BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

from microformer.tokenizer import SPECIAL_TOKENS, BPETokenizer

# ---- Helpers ---------------------------------------------------------------

_CORPUS = "aaabdaaabac" * 50  # Repetitive text with clear merge candidates.


def _trained_tokenizer(num_merges: int = 10) -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(_CORPUS, num_merges=num_merges)
    return tok


# ===========================================================================
# Roundtrip encode / decode
# ===========================================================================


class TestRoundtrip:
    def test_ascii_roundtrip(self) -> None:
        tok = _trained_tokenizer()
        text = "aaabdaaabac"
        assert tok.decode(tok.encode(text)) == text

    def test_unicode_roundtrip(self) -> None:
        """Multi-byte UTF-8 characters survive encode→decode."""
        tok = _trained_tokenizer()
        text = "café 日本語 🚀"
        assert tok.decode(tok.encode(text)) == text

    def test_long_text_roundtrip(self) -> None:
        tok = _trained_tokenizer(num_merges=20)
        text = _CORPUS
        assert tok.decode(tok.encode(text)) == text

    def test_untrained_roundtrip(self) -> None:
        """Even with zero merges, byte-level encoding is lossless."""
        tok = BPETokenizer()
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text


# ===========================================================================
# Special tokens
# ===========================================================================


class TestSpecialTokens:
    def test_endoftext_encodes_to_id_0(self) -> None:
        tok = BPETokenizer()
        ids = tok.encode("<|endoftext|>")
        assert ids == [SPECIAL_TOKENS["<|endoftext|>"]]

    def test_pad_encodes_to_id_1(self) -> None:
        tok = BPETokenizer()
        ids = tok.encode("<|pad|>")
        assert ids == [SPECIAL_TOKENS["<|pad|>"]]

    def test_unk_encodes_to_id_2(self) -> None:
        tok = BPETokenizer()
        ids = tok.encode("<|unk|>")
        assert ids == [SPECIAL_TOKENS["<|unk|>"]]

    def test_special_token_in_context(self) -> None:
        tok = _trained_tokenizer()
        text = "hello<|endoftext|>world"
        ids = tok.encode(text)
        assert SPECIAL_TOKENS["<|endoftext|>"] in ids
        assert tok.decode(ids) == text

    def test_multiple_special_tokens(self) -> None:
        tok = BPETokenizer()
        text = "<|pad|>foo<|endoftext|>"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_decode_special_token_ids(self) -> None:
        tok = BPETokenizer()
        decoded = tok.decode([0, 1, 2])
        assert decoded == "<|endoftext|><|pad|><|unk|>"


# ===========================================================================
# Save / load
# ===========================================================================


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        tok = _trained_tokenizer()
        path = tmp_path / "vocab.json"
        tok.save(path)

        tok2 = BPETokenizer.load(path)
        assert tok2.merge_list == tok.merge_list
        assert tok2.vocab_size == tok.vocab_size

    def test_encode_matches_after_load(self, tmp_path: Path) -> None:
        tok = _trained_tokenizer()
        path = tmp_path / "vocab.json"
        tok.save(path)
        tok2 = BPETokenizer.load(path)

        text = "aaabdaaabac café"
        assert tok2.encode(text) == tok.encode(text)

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        tok = _trained_tokenizer()
        path = tmp_path / "vocab.json"
        tok.save(path)

        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "special_tokens" in data
        assert "merges" in data
        assert isinstance(data["merges"], list)


# ===========================================================================
# Unknown / invalid bytes
# ===========================================================================


class TestUnknownBytes:
    def test_decode_unknown_id_gives_unk_marker(self) -> None:
        tok = BPETokenizer()
        result = tok.decode([99999])
        assert "<|unk|>" in result

    def test_invalid_utf8_decoded_with_replacement(self) -> None:
        """Raw byte ids that form invalid UTF-8 should not crash."""
        tok = BPETokenizer()
        # 0xFF is not valid start of a UTF-8 sequence.
        byte_id = 3 + 0xFF  # _BYTE_OFFSET + 0xFF
        result = tok.decode([byte_id])
        # Should contain the Unicode replacement character.
        assert "\ufffd" in result


# ===========================================================================
# Empty string
# ===========================================================================


class TestEmptyString:
    def test_encode_empty(self) -> None:
        tok = BPETokenizer()
        assert tok.encode("") == []

    def test_decode_empty(self) -> None:
        tok = BPETokenizer()
        assert tok.decode([]) == ""

    def test_roundtrip_empty(self) -> None:
        tok = BPETokenizer()
        assert tok.decode(tok.encode("")) == ""


# ===========================================================================
# Merge priority correctness
# ===========================================================================


class TestMergePriority:
    def test_most_frequent_pair_merged_first(self) -> None:
        """The first merge should be the most common byte pair."""
        tok = BPETokenizer()
        # "aa" appears 3 times, "ab" only once — "aa" should merge first.
        tok.train("aaa ab", num_merges=1)
        assert len(tok.merge_list) == 1
        # The merged pair should be ('a', 'a') as byte ids.
        a_id = 3 + ord("a")
        assert tok.merge_list[0] == (a_id, a_id)

    def test_merges_reduce_token_count(self) -> None:
        """Each merge should reduce the encoded length."""
        tok0 = BPETokenizer()  # No merges.
        tok5 = _trained_tokenizer(num_merges=5)

        text = _CORPUS
        len0 = len(tok0.encode(text))
        len5 = len(tok5.encode(text))
        assert len5 < len0

    def test_merge_order_preserved(self) -> None:
        """Earlier merges have higher priority than later ones."""
        tok = _trained_tokenizer(num_merges=5)
        # First merge id should be 259 (_BYTE_OFFSET + 256 + 0).
        first_merge_id = 3 + 256  # 259
        assert tok.merge_to_id[tok.merge_list[0]] == first_merge_id

    def test_vocab_size_grows_with_merges(self) -> None:
        tok0 = BPETokenizer()
        tok5 = _trained_tokenizer(num_merges=5)
        # Base: 3 special + 256 bytes = 259.
        assert tok0.vocab_size == 259
        assert tok5.vocab_size == 259 + 5
