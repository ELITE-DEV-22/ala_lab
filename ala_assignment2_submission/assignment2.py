import math
import numbers
import string
from pathlib import Path
from typing import Self

import numpy as np

class Vec:
    """A real-valued vector stored as a tuple."""

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = tuple()
        else:
            elements = tuple(src)

            for x in elements:
                if not isinstance(x, numbers.Real):
                    raise TypeError(f"Scalar must be a real number: {type(x)}")

            self.elements = tuple(float(x) for x in elements)

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def dot(self, other: Self) -> float:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        if len(self) != len(other):
            raise ValueError("Vectors must be of same dimensions")

        return sum(
            x * y
            for x, y in zip(self.elements, other.elements)
        )

    def norm(self) -> float:
        return math.sqrt(
            sum(x * x for x in self.elements)
        )

    def cosine_similarity(self, other: Self) -> float:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        self_norm = self.norm()
        other_norm = other.norm()

        if self_norm == 0 or other_norm == 0:
            raise ValueError(
                "Cosine similarity is undefined for a zero vector"
            )

        return self.dot(other) / (self_norm * other_norm)


TAGS = [
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "technology",
    "curriculum",
    "collaboration",
    "publication",
    "laboratory",
    "scholarship",
    "mentorship",
    "internship",
    "entrepreneurship",
    "accreditation",
    "alumni",
]

assert len(TAGS) == 20


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
    # Additional stopwords included for the supplied text:
    "it", "this", "that", "our", "we",
}


def preprocess_text(text, model):
    """Apply the six preprocessing steps specified in Part C."""

    # Step 1: Convert to lowercase.
    text = text.lower()

    # Step 2: Remove punctuation.
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Step 3: Split into tokens.
    tokens = text.split()

    # Step 4: Remove stopwords.
    tokens = [
        token for token in tokens
        if token not in STOPWORDS
    ]

    # Step 5: Remove tokens whose length is <= 2.
    tokens = [
        token for token in tokens
        if len(token) > 2
    ]

    # Step 6: Separate in-vocabulary and OOV tokens.
    in_vocab_tokens = []
    oov_tokens = []

    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)

    return tokens, in_vocab_tokens, oov_tokens


def build_tag_matrix(model, tags):
    """Build T with shape (20, 50)."""

    if len(tags) != 20:
        raise ValueError("Exactly 20 tags are required.")

    tag_vectors = []

    for tag in tags:
        words = tag.lower().split()

        # Every component word must be in the GloVe vocabulary.
        for word in words:
            if word not in model.key_to_index:
                raise ValueError(f"Tag word not in vocabulary: {word}")

        # A single-word tag uses its word vector.
        # A multi-word tag uses the mean of its component vectors.
        word_vectors = [model[word] for word in words]
        tag_vector = np.mean(word_vectors, axis=0)
        tag_vectors.append(tag_vector)

    T = np.vstack(tag_vectors)
    assert T.shape == (20, 50)

    return list(tags), T


def build_text_matrix(model, tokens):
    """Build W with shape (n, 50), preserving token order and repetitions."""

    in_vocab_tokens = []
    oov_tokens = []

    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)

    if len(in_vocab_tokens) == 0:
        raise ValueError(
            "No in-vocabulary tokens found in the input text."
        )

    W = np.vstack([
        model[token] for token in in_vocab_tokens
    ])

    assert W.shape == (len(in_vocab_tokens), 50)

    return in_vocab_tokens, oov_tokens, W

def similarity_matrix_vector_class(W, T):
    """Compute all word/tag cosine similarities using Vec and Python loops."""

    if W.ndim != 2 or W.shape[1] != 50:
        raise ValueError("W must have shape (n_words, 50).")

    if T.shape != (20, 50):
        raise ValueError("T must have shape (20, 50).")

    n_words = W.shape[0]
    S_vector = np.zeros((n_words, 20))

    for i in range(n_words):
        for j in range(20):
            word_vector = Vec(W[i])
            tag_vector = Vec(T[j])
            S_vector[i, j] = word_vector.cosine_similarity(tag_vector)

    assert S_vector.shape == (n_words, 20)
    return S_vector


def similarity_matrix_numpy(W, T):
    """Compute all word/tag cosine similarities using NumPy operations."""

    if W.ndim != 2 or W.shape[1] != 50:
        raise ValueError("W must have shape (n_words, 50).")

    if T.shape != (20, 50):
        raise ValueError("T must have shape (20, 50).")

    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)

    # Check row norms before division.
    if np.any(W_norm == 0) or np.any(T_norm == 0):
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    W_hat = W / W_norm
    T_hat = T / T_norm

    S_numpy = W_hat @ T_hat.T

    assert S_numpy.shape == (W.shape[0], 20)
    return S_numpy





def verify_similarity_matrices(S_vector, S_numpy):
    """Return allclose result and maximum absolute difference."""

    S_vector = np.asarray(S_vector)
    S_numpy = np.asarray(S_numpy)

    are_equal = np.allclose(
        S_vector,
        S_numpy,
        rtol=1e-5,
        atol=1e-8,
    )
    max_difference = np.max(np.abs(S_vector - S_numpy))

    return are_equal, max_difference

def rank_tags(tag_names, S, in_vocab_tokens):
    """Rank tags by max-pool score and record each best-matching word."""

    if S.shape != (len(in_vocab_tokens), len(tag_names)):
        raise ValueError("S shape must match the tokens and tag names.")

    ranked_tags = []

    for j, tag in enumerate(tag_names):
        best_word_index = int(np.argmax(S[:, j]))
        best_word = in_vocab_tokens[best_word_index]
        score = float(S[best_word_index, j])

        if best_word == tag.lower():
            match_type = "Identical word"
        else:
            match_type = "Different word"

        ranked_tags.append((tag, score, best_word, match_type))

    ranked_tags.sort(key=lambda item: item[1], reverse=True)
    return ranked_tags


def print_top_eight(ranked_tags):
    print("\nTop 8 Ranked Tags")
    print("-" * 75)
    print(
        f"{'Rank':<6}{'Tag':<20}{'Score':<12}"
        f"{'Best-matching word':<23}{'Match type'}"
    )

    for rank, (tag, score, best_word, match_type) in enumerate(
        ranked_tags[:8], start=1
    ):
        print(
            f"{rank:<6}{tag:<20}{score:<12.4f}"
            f"{best_word:<23}{match_type}"
        )

def main():
    # Load the required pretrained GloVe model.
    import gensim.downloader as api

    model = api.load("glove-wiki-gigaword-50")

    # Part B - Load and check the input text file.
    text_file = Path(__file__).resolve().parent / "manipal_text.txt"

    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()

    tokens_before = len(text.split())

    if tokens_before > 400:
        raise ValueError(
            f"Input text contains {tokens_before} words. "
            "The maximum allowed is 400."
        )

    print("Input text loaded successfully.")
    print("Word count:", tokens_before)

    # Part C - Preprocess the text and report token statistics.
    tokens, preprocessed_in_vocab, preprocessed_oov = preprocess_text(
        text, model
    )

    print("\nPart C - Token Statistics")
    print("Tokens before preprocessing:", tokens_before)
    print("Tokens after preprocessing:", len(tokens))
    print("In-vocabulary tokens:", len(preprocessed_in_vocab))
    print("OOV tokens:", len(preprocessed_oov))
    print("Distinct OOV count:", len(set(preprocessed_oov)))
    print("Distinct OOV tokens:", sorted(set(preprocessed_oov)))

    # Part D - Construct T and W.
    tag_names, T = build_tag_matrix(model, TAGS)
    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, tokens)

    assert in_vocab_tokens == preprocessed_in_vocab
    assert oov_tokens == preprocessed_oov
    assert T.shape == (20, 50)
    assert W.shape == (len(in_vocab_tokens), 50)

    print("\nPart D - Matrix Shapes")
    print("Tag matrix shape:", T.shape)
    print("Text matrix shape:", W.shape)

    # Part E - Similarities using the Vec class.
    S_vector = similarity_matrix_vector_class(W, T)
    print("\nVector-class similarity matrix shape:", S_vector.shape)

    # Part F - Similarities using NumPy.
    S_numpy = similarity_matrix_numpy(W, T)
    print("NumPy similarity matrix shape:", S_numpy.shape)

    # Part G - Compare both matrices.
    are_equal, max_difference = verify_similarity_matrices(
        S_vector, S_numpy
    )
    print("Are both similarity matrices equal?", are_equal)
    print("Maximum absolute difference:", max_difference)

    # Part H - Rank the tags using max pooling.
    ranked_tags = rank_tags(tag_names, S_numpy, in_vocab_tokens)
    print_top_eight(ranked_tags)


if __name__ == "__main__":
    main()
