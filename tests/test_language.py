from genesis.language.segmentation import segment_text
from genesis.language.tokenizer import CharTokenizer
from genesis.language.vocabulary import Vocabulary


def test_segmentation_splits_words_and_punctuation():
    assert segment_text("A cat, sitting.") == ["A", "cat", ",", "sitting", "."]


def test_vocabulary_is_deterministic_and_round_trips(tmp_path):
    tokenizer = CharTokenizer.train(["aba", "abc", "cab"])
    assert "a" in tokenizer.vocabulary
    ids = tokenizer.encode("cab", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "cab"

    path = tmp_path / "vocab.json"
    tokenizer.vocabulary.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.to_dict() == tokenizer.vocabulary.to_dict()


def test_vocabulary_unknown_token_maps_to_unk():
    vocab = Vocabulary.build([["a", "b"]])
    ids = vocab.encode(["a", "zzz"])
    assert ids[0] == vocab.token_to_id("a")
    assert ids[1] == vocab.unk_id


def test_vocabulary_min_frequency():
    vocab = Vocabulary.build([["a", "a", "b"]], min_frequency=2)
    assert "a" in vocab
    assert "b" not in vocab
