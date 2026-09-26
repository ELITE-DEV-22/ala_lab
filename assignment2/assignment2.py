import math
from typing import Self
import gensim.downloader as api
import string

class Vec:

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = []
        else:
            elements = list(src)

            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")

            self.elements = elements

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

model = api.load("glove-wiki-gigaword-50")

# Part A - 20 tags
tags = [
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
    "alumni"
]


# Verify that every tag is in the model vocabulary
for tag in tags:
    if tag not in model.key_to_index:
        print("Tag not in vocabulary:", tag)


STOPWORDS ={
     "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from"
}

def preprocess_text(text, model):
    #convert text to lower case
    text = text.lower()

    #remove puntuation
    text = text.translate(
        str.maketrans("", "", string.punctuation))

    #split into tokens
    tokens = text.split()

    #remove stopwords
    tokens = [
        token for token in tokens
        if token not in STOPWORDS
    ]

    #remove tokens with lenght less than 2
    tokens =[
        token for token in tokens
        if len(token) > 2
    ]

    #check glove vocabulary
    in_vocab_tokens =[]
    oov_tokens = []

    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)
    return tokens, in_vocab_tokens, oov_tokens

with open("manipal_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokens, in_vocab_tokens, oov_tokens = preprocess_text(text, model)

print("Tokens after preprocessing:", len(tokens))
print("In-vocabulary tokens:", len(in_vocab_tokens))
print("OOV tokens:", len(oov_tokens))
print("Distinct OOV tokens:", len(set(oov_tokens)))