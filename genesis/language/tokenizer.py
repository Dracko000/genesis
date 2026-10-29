"""Character-level tokenizer built and documented by the project.

This is the initial language baseline: transparent, deterministic, and with no
pretrained semantics. It tokenizes by Unicode code point, uses a
:class:`~genesis.language.vocabulary.Vocabulary` built from observed text, and
round-trips losslessly for in-vocabulary input.

Tokenization does not provide semantic understanding; it only maps text to
inspectable symbols.
"""

from __future__ import annotations

from collections.abc import Iterable

from genesis.language.vocabulary import Vocabulary


class CharTokenizer:
    """A minimal character-level tokenizer."""

    name = "char"

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_size: int | None = None,
    ) -> "CharTokenizer":
        """Build a character vocabulary from a corpus of strings."""

        sequences = (list(text) for text in texts)
        vocabulary = Vocabulary.build(
            sequences, min_frequency=min_frequency, max_size=max_size
        )
        return cls(vocabulary)

    def tokenize(self, text: str) -> list[str]:
        """Return one symbol per character."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self.vocabulary.encode(
            self.tokenize(text), add_bos=add_bos, add_eos=add_eos
        )

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return "".join(self.vocabulary.decode(ids, skip_special=skip_special))

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)
