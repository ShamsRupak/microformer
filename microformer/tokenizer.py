"""Byte-level BPE tokenizer for MicroFormer.

Implements the Byte-Pair Encoding algorithm from scratch:
1. Text → UTF-8 bytes → initial token sequence (one token per byte).
2. Iteratively merge the most frequent adjacent pair into a new token.
3. Encoding applies learned merges greedily in priority order.
4. Decoding maps token ids → byte sequences → UTF-8 text.

No external tokenizer libraries (tiktoken, HuggingFace tokenizers, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 0,
    "<|pad|>": 1,
    "<|unk|>": 2,
}

_NUM_SPECIAL = len(SPECIAL_TOKENS)

# Byte-level base vocabulary: ids 3..258 map to bytes 0..255.
_BYTE_OFFSET = _NUM_SPECIAL


def _byte_to_id(b: int) -> int:
    return b + _BYTE_OFFSET


def _id_to_byte(token_id: int) -> int:
    return token_id - _BYTE_OFFSET


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Byte-level Byte-Pair Encoding tokenizer.

    Vocabulary layout:
        [0..2]      → special tokens  (<|endoftext|>, <|pad|>, <|unk|>)
        [3..258]    → single-byte tokens (0x00..0xFF)
        [259..]     → BPE merge tokens (in merge-order)
    """

    def __init__(self) -> None:
        # merge_list[i] = (id_a, id_b) — the i-th merge, producing
        # token id (_BYTE_OFFSET + 256 + i).
        self.merge_list: list[tuple[int, int]] = []
        # Fast lookup: (id_a, id_b) → merged token id.
        self.merge_to_id: dict[tuple[int, int], int] = {}

        # Reverse mapping: token id → byte sequence (built lazily).
        self._id_to_bytes: dict[int, bytes] = {}
        self._build_base_vocab()

    # ---- Vocabulary bookkeeping -------------------------------------------

    def _build_base_vocab(self) -> None:
        """Populate id→bytes for the 256 single-byte tokens."""
        self._id_to_bytes = {}
        for b in range(256):
            self._id_to_bytes[_byte_to_id(b)] = bytes([b])

    def _rebuild_merge_vocab(self) -> None:
        """Rebuild id→bytes for all merge tokens from merge_list."""
        self._build_base_vocab()
        for i, (a, b) in enumerate(self.merge_list):
            new_id = _BYTE_OFFSET + 256 + i
            self._id_to_bytes[new_id] = self._id_to_bytes[a] + self._id_to_bytes[b]

    @property
    def vocab_size(self) -> int:
        return _NUM_SPECIAL + 256 + len(self.merge_list)

    # ---- Training ---------------------------------------------------------

    def train(self, text: str, num_merges: int) -> None:
        """Learn BPE merges from *text*.

        Args:
            text: Training corpus (plain text).
            num_merges: Number of merge operations to learn.
        """
        # Encode corpus to a mutable list of byte-level token ids.
        token_ids = [_byte_to_id(b) for b in text.encode("utf-8")]

        self.merge_list = []
        self.merge_to_id = {}
        self._build_base_vocab()

        for _ in range(num_merges):
            if len(token_ids) < 2:
                break

            # Count adjacent pairs.
            pair_counts: dict[tuple[int, int], int] = {}
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Pick the most frequent pair (ties broken deterministically
            # by the pair value itself for reproducibility).
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 2:
                # No pair occurs more than once — stop early.
                break

            # Assign a new token id to this merge.
            new_id = _BYTE_OFFSET + 256 + len(self.merge_list)
            self.merge_list.append(best_pair)
            self.merge_to_id[best_pair] = new_id
            self._id_to_bytes[new_id] = (
                self._id_to_bytes[best_pair[0]] + self._id_to_bytes[best_pair[1]]
            )

            # Apply the merge in-place on the training sequence.
            token_ids = self._apply_single_merge(token_ids, best_pair, new_id)

    @staticmethod
    def _apply_single_merge(
        ids: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """Replace every occurrence of *pair* with *new_id*."""
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ---- Encoding ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids.

        Special-token strings embedded in *text* are recognised and mapped
        to their reserved ids.  Everything else goes through byte-level
        BPE.
        """
        if not text:
            return []

        # Split on special tokens while preserving them.
        segments = self._split_on_special_tokens(text)
        token_ids: list[int] = []
        for segment, is_special in segments:
            if is_special:
                token_ids.append(SPECIAL_TOKENS[segment])
            else:
                token_ids.extend(self._encode_bytes(segment))
        return token_ids

    def _encode_bytes(self, text: str) -> list[int]:
        """BPE-encode a plain text segment (no special tokens)."""
        if not text:
            return []
        ids = [_byte_to_id(b) for b in text.encode("utf-8")]

        # Greedily apply merges in priority order.
        for pair, merged_id in self.merge_to_id.items():
            ids = self._apply_single_merge(ids, pair, merged_id)
            if len(ids) < 2:
                break
        return ids

    @staticmethod
    def _split_on_special_tokens(
        text: str,
    ) -> list[tuple[str, bool]]:
        """Split text into (segment, is_special) pairs."""
        segments: list[tuple[str, bool]] = []
        remaining = text
        while remaining:
            earliest_pos = len(remaining)
            earliest_token = ""
            for st in SPECIAL_TOKENS:
                pos = remaining.find(st)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = st
            if not earliest_token:
                segments.append((remaining, False))
                break
            if earliest_pos > 0:
                segments.append((remaining[:earliest_pos], False))
            segments.append((earliest_token, True))
            remaining = remaining[earliest_pos + len(earliest_token) :]
        return segments

    # ---- Decoding ---------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token ids back to a string.

        Special token ids are mapped to their string representations.
        Unknown ids are replaced with the <|unk|> string.
        Byte sequences are decoded as UTF-8 with replacement for
        invalid sequences.
        """
        # Reverse lookup for special tokens.
        id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}

        byte_buffer = bytearray()
        parts: list[str] = []

        def flush_bytes() -> None:
            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for tid in token_ids:
            if tid in id_to_special:
                flush_bytes()
                parts.append(id_to_special[tid])
            elif tid in self._id_to_bytes:
                byte_buffer.extend(self._id_to_bytes[tid])
            else:
                flush_bytes()
                parts.append("<|unk|>")

        flush_bytes()
        return "".join(parts)

    # ---- Persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary and merges to a JSON file."""
        data = {
            "special_tokens": SPECIAL_TOKENS,
            "merges": [[a, b] for a, b in self.merge_list],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merge_list = [tuple(pair) for pair in data["merges"]]
        tok.merge_to_id = {
            tuple(pair): _BYTE_OFFSET + 256 + i for i, pair in enumerate(tok.merge_list)
        }
        tok._rebuild_merge_vocab()
        return tok
