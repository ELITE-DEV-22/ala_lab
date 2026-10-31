# AME 5151 - Applied Linear Algebra Lab - Assignment 2

This folder contains the implementation and supporting files for the Semantic Tagging with GloVe and NumPy assignment.

## Files

- `assignment2.py` - Parts A-H: GloVe loading, tags, preprocessing, matrices T and W, both similarity matrices, numerical verification, and max-pool ranking.
- `test_assignment2.py` - The seven unit tests required by the assignment.
- `manipal_text.txt` - The supplied eight text excerpts (108 whitespace-separated words).
- `linkedin_citations.txt` - LinkedIn post citations corresponding to the eight excerpts.
- `lab_report_answers.md` - Tag-vocabulary justification, stopword documentation, and answers to all nine report questions.
- `requirements.txt` - Python package requirements.

## Setup

Use Python 3.13 (the Gensim package has prebuilt Windows wheels for this Python version).

```powershell
python -m pip install -r requirements.txt
```

The first run downloads the pretrained `glove-wiki-gigaword-50` model through Gensim. An internet connection is needed for that initial download.

## Run the assignment

From this folder:

```powershell
python assignment2.py
```

The program prints the Part C token statistics, the shapes of `T` and `W`, the shapes of both similarity matrices, the `np.allclose` result, the maximum absolute difference, and the top eight ranked tags.

To save the program output for submission:

```powershell
python assignment2.py > program_output.txt 2>&1
```

## Run the seven unit tests

```powershell
python -m unittest -v test_assignment2.py
```

The tests use a small deterministic fake embedding model so they do not need to download GloVe. The actual assignment program uses the required pretrained GloVe model.

## Notes

- Keep `manipal_text.txt` and `assignment2.py` in the same folder.
- Keep LinkedIn URLs in `linkedin_citations.txt`, not in `manipal_text.txt`, because the Python program tokenizes the text file.
- The input text contains 108 whitespace-separated words, below the 400-word limit.
- The code preserves repeated in-vocabulary tokens in `W`.
- Part E uses the custom `Vec` implementation and Python loops. Part F uses NumPy row normalization and matrix multiplication.
