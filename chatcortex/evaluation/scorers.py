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

For pipelines that extract triples, three more scorers compare a set of
extracted triples with a set of reference triples:

- :func:`triple_precision`: share of the extracted triples that are in
  the reference.
- :func:`triple_recall`: share of the reference triples that were
  extracted.
- :func:`triple_f1`: harmonic mean of the two.

Here ``output`` and ``expected`` are each a collection of triples, and a
triple is a sequence of fields such as ``(subject, predicate, object)``.
A triple counts as found only when every field matches after
normalisation, so adding a field (a modality, a source) makes the match
stricter. Duplicates count once. Two empty collections score 1; one
empty collection scores 0.
"""

import re
import string
from collections import Counter
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Iterable, Optional, Set, Tuple, Union

_PUNCTUATION = set(string.punctuation)

Expected = Union[str, Iterable[str]]


@lru_cache(maxsize=65536)
def _normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize(text: Any) -> str:
    # Cached: resampling a metric normalises the same fields thousands of times.
    return _normalize_text(str(text if text is not None else ""))


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


Triples = Optional[Iterable[Iterable[Any]]]


def _triple_set(triples: Triples, name: str) -> Set[Tuple[str, ...]]:
    if triples is None:
        return set()
    if isinstance(triples, (str, bytes)):
        raise TypeError(
            f"{name} must be a collection of triples, a string was given; "
            "parse it into (subject, predicate, object) sequences first"
        )
    found = set()
    for triple in triples:
        # A dict would iterate over its keys, so every dict would match.
        if isinstance(triple, (str, bytes, Mapping)) or not isinstance(triple, IterableABC):
            raise TypeError(
                f"each triple in {name} must be a sequence of fields such as "
                f"(subject, predicate, object), {triple!r} given"
            )
        found.add(tuple(normalize(field) for field in triple))
    return found


def _triple_scores(output: Triples, expected: Triples) -> Tuple[float, float]:
    """Precision and recall of ``output`` against ``expected``."""
    predicted = _triple_set(output, "output")
    reference = _triple_set(expected, "expected")
    if not predicted and not reference:
        return 1.0, 1.0
    if not predicted or not reference:
        return 0.0, 0.0
    matched = len(predicted & reference)
    return matched / len(predicted), matched / len(reference)


def triple_precision(output: Triples, expected: Triples) -> float:
    return _triple_scores(output, expected)[0]


def triple_recall(output: Triples, expected: Triples) -> float:
    return _triple_scores(output, expected)[1]


def triple_f1(output: Triples, expected: Triples) -> float:
    precision, recall = _triple_scores(output, expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
