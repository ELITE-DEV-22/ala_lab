import unittest

import numpy as np

from assignment2 import (
    TAGS,
    Vec,
    build_tag_matrix,
    build_text_matrix,
    similarity_matrix_vector_class,
    similarity_matrix_numpy,
)


class FakeModel:
    """Small deterministic stand-in for GloVe, used only by unit tests."""

    def __init__(self):
        rng = np.random.default_rng(42)
        self.vectors = {
            word: rng.normal(size=50).astype(np.float32)
            for word in TAGS
        }
        self.key_to_index = {
            word: index for index, word in enumerate(self.vectors)
        }

    def __getitem__(self, word):
        return self.vectors[word]


class TestAssignment2(unittest.TestCase):

    def setUp(self):
        self.model = FakeModel()
        self.tag_names, self.T = build_tag_matrix(self.model, TAGS)
        self.tokens = ["research", "innovation", "research"]
        self.in_vocab_tokens, self.oov_tokens, self.W = build_text_matrix(
            self.model, self.tokens
        )
        self.S_vector = similarity_matrix_vector_class(self.W, self.T)
        self.S_numpy = similarity_matrix_numpy(self.W, self.T)

    def test_tag_matrix_shape(self):
        self.assertEqual(self.T.shape, (20, 50))

    def test_text_matrix_shape(self):
        self.assertEqual(
            self.W.shape,
            (len(self.in_vocab_tokens), 50),
        )

    def test_similarity_matrix_shape(self):
        self.assertEqual(
            self.S_numpy.shape,
            (len(self.in_vocab_tokens), 20),
        )

    def test_single_cosine_similarity(self):
        vector_result = Vec(self.W[0]).cosine_similarity(
            Vec(self.T[0])
        )
        numpy_result = self.S_numpy[0, 0]
        self.assertAlmostEqual(vector_result, float(numpy_result), places=6)

    def test_similarity_matrices_are_equal(self):
        self.assertTrue(
            np.allclose(
                self.S_vector,
                self.S_numpy,
                rtol=1e-5,
                atol=1e-8,
            )
        )

    def test_oov_text_token_is_reported_and_excluded(self):
        in_vocab, oov, W = build_text_matrix(
            self.model,
            ["research", "not_in_glove"],
        )
        self.assertEqual(in_vocab, ["research"])
        self.assertEqual(oov, ["not_in_glove"])
        self.assertEqual(W.shape, (1, 50))

    def test_oov_tag_component_raises_error(self):
        invalid_tags = list(TAGS)
        invalid_tags[0] = "not_in_glove"

        with self.assertRaises(ValueError):
            build_tag_matrix(self.model, invalid_tags)


if __name__ == "__main__":
    unittest.main(verbosity=2)
