"""Ready-made scorers for :class:`~chatcortex.evaluation.eval_set.EvalSet`.

Each takes ``(output, expected)`` and returns a score in ``[0, 1]``.
``expected`` may be one reference answer or a list of acceptable ones;
the best match counts.

- :func:`exact_match`: 1 when the normalised output equals a reference.
- :func:`contains_answer`: 1 when a normalised reference appears in the
  normalised output. Lenient: fine for checking a fact was stated.
- :func:`token_f1`: word overlap between output and reference, as in the
  SQuAD evaluation. Gives partial credit and penalises padding.

Normalisation lower-cases, removes punctuation and the articles "a",
"an" and "the", and collapses whitespace.
"""

import re
import string
from collections import Counter
from typing import Any, Iterable, Union

_PUNCTUATION = set(string.punctuation)

Expected = Union[str, Iterable[str]]


def normalize(text: Any) -> str:
    text = str(text if text is not None else "").lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _references(expected: Expected) -> list:
    if isinstance(expected, str):
        return [expected]
    references = list(expected)
    if not references:
        raise ValueError("expected must hold at least one reference answer")
    return references


def exact_match(output: Any, expected: Expected) -> float:
    predicted = normalize(output)
    return float(any(predicted == normalize(ref) for ref in _references(expected)))


def contains_answer(output: Any, expected: Expected) -> float:
    predicted = normalize(output)
    return float(
        any(normalize(ref) and normalize(ref) in predicted for ref in _references(expected))
    )


def _f1(predicted: str, reference: str) -> float:
    predicted_words = predicted.split()
    reference_words = reference.split()
    if not predicted_words or not reference_words:
        return float(predicted_words == reference_words)
    overlap = sum((Counter(predicted_words) & Counter(reference_words)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_words)
    recall = overlap / len(reference_words)
    return 2 * precision * recall / (precision + recall)


def token_f1(output: Any, expected: Expected) -> float:
    predicted = normalize(output)
    return max(_f1(predicted, normalize(ref)) for ref in _references(expected))
