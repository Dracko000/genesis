"""A small, buildable token vocabulary.

The vocabulary maps tokens (characters or words) to integer ids and back. It
supports special tokens, frequency filtering, deterministic ordering and JSON
round-tripping. It has no pretrained embeddings and no hidden semantic state.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
DEFAULT_SPECIALS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class Vocabulary:
    """Deterministic token <-> id mapping."""

    def __init__(self, tokens: Iterable[str] = (), *, specials: Iterable[str] = DEFAULT_SPECIALS):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        self.specials = tuple(dict.fromkeys(specials))
        for token in self.specials:
            self._add(token)
        for token in tokens:
            self._add(token)

    def _add(self, token: str) -> None:
        if token in self._token_to_id:
            return
        self._token_to_id[token] = len(self._id_to_token)
        self._id_to_token.append(token)

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Iterable[str]],
        *,
        min_frequency: int = 1,
        max_size: int | None = None,
        specials: Iterable[str] = DEFAULT_SPECIALS,
    ) -> "Vocabulary":
        """Build from tokenised sequences, keeping frequent tokens first.

        Ties are broken by the token string so that the resulting vocabulary is
        independent of input ordering given the same counts.
        """

        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        counts: Counter[str] = Counter()
        for sequence in token_sequences:
            counts.update(sequence)
        candidates = [
            token
            for token, count in counts.items()
            if count >= min_frequency and token not in specials
        ]
        candidates.sort(key=lambda token: (-counts[token], token))
        if max_size is not None:
            if max_size < len(tuple(specials)):
                raise ValueError("max_size must be >= number of special tokens")
            candidates = candidates[: max_size - len(tuple(specials))]
        return cls(candidates, specials=specials)

    def __len__(self) -> int:
        return len(self._id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    @property
    def pad_id(self) -> int:
        return self.token_to_id(PAD_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.token_to_id(UNK_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_to_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_to_id(EOS_TOKEN)

    def token_to_id(self, token: str) -> int:
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise KeyError(f"token {token!r} is not in the vocabulary") from exc

    def id_to_token(self, token_id: int) -> str:
        try:
            return self._id_to_token[token_id]
        except (IndexError, TypeError) as exc:
            raise KeyError(f"token id {token_id!r} is not in the vocabulary") from exc

    def encode(self, tokens: Iterable[str], *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        encoded: list[int] = []
        if add_bos:
            encoded.append(self.bos_id)
        unk = self.unk_id
        encoded.extend(self._token_to_id.get(token, unk) for token in tokens)
        if add_eos:
            encoded.append(self.eos_id)
        return encoded

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> list[str]:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token(token_id)
            if skip_special and token in self.specials:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {"specials": list(self.specials), "tokens": list(self._id_to_token)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Vocabulary":
        tokens = data.get("tokens", [])
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("invalid vocabulary file: 'tokens' must be a list of strings")
        specials = data.get("specials", list(DEFAULT_SPECIALS))
        if not isinstance(specials, list) or not all(isinstance(token, str) for token in specials):
            raise ValueError("invalid vocabulary file: 'specials' must be a list of strings")
        vocab = cls(tokens, specials=specials)
        if vocab._id_to_token != tokens:
            raise ValueError("invalid vocabulary file: token order is inconsistent")
        return vocab

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
