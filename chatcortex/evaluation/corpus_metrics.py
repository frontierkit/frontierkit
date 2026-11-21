"""Ready-made whole-set metrics for :class:`~chatcortex.evaluation.eval_set.EvalSet`.

A scorer grades one output; a metric here takes every output and every
reference at once, as ``metric(outputs, expecteds)``, and returns one
number for the whole set. Pass them as ``EvalSet(..., metrics={...})``.

All of them expect each output and each reference to be a collection of
triples, as the triple scorers do: a triple is a sequence of fields,
compared field by field after normalisation, and duplicates count once.
A failed run's output is ``None`` and counts as no triples. A triple is
only ever matched against the reference of its own example.

- :func:`pooled_triple_precision`, :func:`pooled_triple_recall` and
  :func:`pooled_triple_f1` count matches over all examples together, so
  an example with many triples weighs more than one with few. The mean
  of the per-example scorers weighs every example the same.
- :func:`label_macro_f1` treats one field of each triple as its label (a
  modality, a relation type) and averages the F1 of each label, so a
  rare label weighs as much as a common one.
"""

from typing import Any, Dict, List, Sequence, Set, Tuple

from .scorers import _triple_set

Item = Tuple[int, Tuple[str, ...]]


def _paired_sets(
    outputs: Sequence[Any], expecteds: Sequence[Any]
) -> List[Tuple[Set[Tuple[str, ...]], Set[Tuple[str, ...]]]]:
    if len(outputs) != len(expecteds):
        raise ValueError(
            f"outputs and expecteds must be the same length, "
            f"{len(outputs)} and {len(expecteds)} given"
        )
    return [
        (_triple_set(output, "output"), _triple_set(expected, "expected"))
        for output, expected in zip(outputs, expecteds)
    ]


def _pooled_scores(outputs: Sequence[Any], expecteds: Sequence[Any]) -> Tuple[float, float]:
    """Precision and recall with counts summed over all examples."""
    matched = predicted = reference = 0
    for found, wanted in _paired_sets(outputs, expecteds):
        matched += len(found & wanted)
        predicted += len(found)
        reference += len(wanted)
    if not predicted and not reference:
        return 1.0, 1.0
    if not predicted or not reference:
        return 0.0, 0.0
    return matched / predicted, matched / reference


def pooled_triple_precision(outputs: Sequence[Any], expecteds: Sequence[Any]) -> float:
    return _pooled_scores(outputs, expecteds)[0]


def pooled_triple_recall(outputs: Sequence[Any], expecteds: Sequence[Any]) -> float:
    return _pooled_scores(outputs, expecteds)[1]


def pooled_triple_f1(outputs: Sequence[Any], expecteds: Sequence[Any]) -> float:
    precision, recall = _pooled_scores(outputs, expecteds)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _by_label(
    triples: Set[Tuple[str, ...]], example: int, field: int, name: str
) -> Dict[str, Set[Item]]:
    """Group one example's triples by label, keyed by the remaining fields."""
    groups: Dict[str, Set[Item]] = {}
    for triple in triples:
        try:
            label = triple[field]
        except IndexError:
            raise ValueError(
                f"a triple in {name} has {len(triple)} fields, so it has no "
                f"label at position {field}: {triple!r}"
            ) from None
        position = field % len(triple)
        rest = triple[:position] + triple[position + 1:]
        groups.setdefault(label, set()).add((example, rest))
    return groups


def label_macro_f1(
    outputs: Sequence[Any],
    expecteds: Sequence[Any],
    field: int = -1,
    matched_only: bool = False,
) -> float:
    """F1 of each label, averaged over the labels with equal weight.

    One field of each triple is its label; the other fields identify the
    triple. A labelled triple is correct when the same triple carries the
    same label in the reference of the same example.

    Args:
        field: Position of the label in each triple; the last by default.
        matched_only: By default every triple counts, so a triple that
            was missed or invented lowers the F1 of its label: the score
            covers extraction and labelling together. With ``True`` only
            triples found in both the output and the reference count,
            whatever their labels, so the score covers labelling alone.

    Labels are those seen in the outputs or the references that count. No
    triples at all scores 1; triples but nothing that counts scores 0.

    Raises:
        ValueError: If a triple is too short to have the label field.
    """
    predicted: Dict[str, Set[Item]] = {}
    reference: Dict[str, Set[Item]] = {}
    for example, (found, wanted) in enumerate(_paired_sets(outputs, expecteds)):
        for label, items in _by_label(found, example, field, "output").items():
            predicted.setdefault(label, set()).update(items)
        for label, items in _by_label(wanted, example, field, "expected").items():
            reference.setdefault(label, set()).update(items)

    if not predicted and not reference:
        return 1.0

    if matched_only:
        in_both = set().union(*predicted.values()) & set().union(*reference.values())
        predicted = {label: items & in_both for label, items in predicted.items()}
        reference = {label: items & in_both for label, items in reference.items()}

    f1_per_label = []
    for label in set(predicted) | set(reference):
        found = predicted.get(label, set())
        wanted = reference.get(label, set())
        if not found and not wanted:
            continue
        correct = len(found & wanted)
        f1_per_label.append(2 * correct / (len(found) + len(wanted)))

    if not f1_per_label:
        return 0.0
    return sum(f1_per_label) / len(f1_per_label)
