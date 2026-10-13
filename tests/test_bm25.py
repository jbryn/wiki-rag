import numpy as np

from wiki_rag.bm25 import BM25, tokenize

DOCUMENTS = [
    "Kant wrote the Critique of Pure Reason",
    "Hume wrote about causation and induction",
    "Causation, causation, causation: a study of cause",
    "A long text that mentions causation once among many other unrelated words about nothing",
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Kant's 'Critique', 1781!") == ["kant", "s", "critique", "1781"]


def test_scores_are_zero_without_matching_terms():
    assert not BM25(DOCUMENTS).scores("Wittgenstein").any()


def test_rare_terms_outweigh_common_ones():
    scores = BM25(DOCUMENTS).scores("Hume causation")

    assert int(np.argmax(scores)) == 1


def test_term_frequency_and_length_affect_ranking():
    scores = BM25(DOCUMENTS).scores("causation")

    assert scores[2] > scores[1] > scores[3] > scores[0] == 0


def test_empty_corpus():
    assert BM25([]).scores("anything").shape == (0,)
