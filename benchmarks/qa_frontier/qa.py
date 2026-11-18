"""Shared pieces of the question-answering frontier benchmark.

The benchmark asks: given a pipeline with several interchangeable parts,
how many model calls does it take to find the pipelines worth choosing
between? Each pipeline has three steps:

1. ``context``: how much reading material the model gets (none, the best
   matching paragraph, or the best three), chosen by simple word overlap.
2. ``answer``: which model answers (free endpoints only).
3. ``postprocess``: keep the reply as it is, or cut it to its first clause.

Only the answer step calls a model. Every reply is stored in a response
cache keyed by model and prompt, so a model is never asked the same thing
twice, collection can resume after a rate limit, and the analysis runs
offline from the cache.

Answers are scored with the SQuAD token F1: the overlap between the reply
and the best-matching reference answer, after the standard normalisation.
"""

import hashlib
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).parent
SAMPLE_DATA = HERE / "sample.json"
DEFAULT_CACHE = HERE / "cache.jsonl"

# --- questions -----------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answers: Tuple[str, ...]
    context: str  # the paragraph it came from; feeds the retrieval pool


def load_questions(
    path: Path, limit: Optional[int] = None, seed: int = 0
) -> List[Question]:
    """Read a SQuAD-format file; optionally sample ``limit`` questions.

    Unanswerable questions (SQuAD 2.0) are skipped. The sample is
    deterministic for a given ``seed`` and returned sorted by id.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers = tuple(dict.fromkeys(a["text"] for a in qa.get("answers", [])))
                if answers:
                    questions.append(
                        Question(qa["id"], qa["question"], answers, paragraph["context"])
                    )
    questions.sort(key=lambda q: q.id)
    if limit is not None and limit < len(questions):
        questions = random.Random(seed).sample(questions, limit)
        questions.sort(key=lambda q: q.id)
    return questions


# --- context: retrieval over the sampled paragraphs -----------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of in on at to for is was were are be been what which who whom "
    "whose when where why how did does do by with from and or as that this it "
    "its into than then there their they he she his her".split()
)


def _terms(text: str) -> List[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


class Retriever:
    """Ranks paragraphs by how many distinct question words they contain."""

    def __init__(self, paragraphs: Iterable[str]):
        self.paragraphs = list(dict.fromkeys(paragraphs))
        self._vocab = [set(_terms(p)) for p in self.paragraphs]

    def top(self, question: str, k: int) -> List[str]:
        words = set(_terms(question))
        ranked = sorted(
            range(len(self.paragraphs)),
            key=lambda i: (-len(words & self._vocab[i]), i),
        )
        return [self.paragraphs[i] for i in ranked[:k]]


INSTRUCTION = "Answer the question with the shortest possible phrase. Do not explain."


def build_prompt(question: str, passages: Sequence[str]) -> str:
    if not passages:
        return f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"
    context = "\n\n".join(passages)
    return (
        f"{INSTRUCTION} Use the context.\n\nContext:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


# name -> number of paragraphs given to the model
CONTEXT_VARIANTS: Dict[str, int] = {"no_context": 0, "top1": 1, "top3": 3}


def context_step(retriever: Retriever, k: int) -> Callable[[str], str]:
    """The context step: question text in, full prompt out."""

    def step(question: str) -> str:
        return build_prompt(question, retriever.top(question, k) if k else [])

    return step


# --- postprocess ---------------------------------------------------------


def as_is(text: Any) -> str:
    return str(text or "").strip()


def first_clause(text: Any) -> str:
    stripped = as_is(text)
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    return re.split(r"(?<=[.;])\s", first_line, maxsplit=1)[0].rstrip(".;")


POSTPROCESS_VARIANTS: Dict[str, Callable[[Any], str]] = {
    "as_is": as_is,
    "first_clause": first_clause,
}

# --- scoring: the SQuAD v1.1 evaluation rules ----------------------------

_PUNCTUATION = set(string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize(prediction).split()
    expected = normalize(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score(prediction: Any, answers: Sequence[str]) -> float:
    """Best F1 against any reference answer, in ``[0, 1]``."""
    return max(token_f1(str(prediction), answer) for answer in answers)


# --- models: free endpoints only -----------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str  # component name; also the cache key prefix, keep it stable
    provider: str
    key_var: str
    base_url: str
    model: str
    declared_latency_ms: float  # a rough prior, before anything is measured


FREE_MODELS: List[ModelSpec] = [
    ModelSpec(
        "saia_llama_8b", "saia", "SAIA_API_KEY",
        "https://chat-ai.academiccloud.de/v1", "meta-llama-3.1-8b-instruct", 800,
    ),
    ModelSpec(
        "saia_gemma_31b", "saia", "SAIA_API_KEY",
        "https://chat-ai.academiccloud.de/v1", "gemma-4-31b-it", 1500,
    ),
    ModelSpec(
        "nvidia_gemma_31b", "nvidia", "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1", "google/gemma-4-31b-it", 5000,
    ),
    ModelSpec(
        "groq_llama_70b", "groq", "GROQ_API_KEY",
        "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", 600,
    ),
    ModelSpec(
        "openrouter_gemma_31b", "openrouter", "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1", "google/gemma-4-31b-it:free", 2000,
    ),
]

# Seconds between calls to one provider, to stay under free-tier limits:
# SAIA's documentation shows 10 requests a minute in its example, Groq
# lists 30 a minute, OpenRouter's free models allow 20 a minute.
PROVIDER_MIN_INTERVAL_S: Dict[str, float] = {
    "saia": 6.5,
    "nvidia": 2.0,
    "groq": 2.5,
    "openrouter": 3.5,
}

REQUEST_OPTIONS = {"max_tokens": 300, "temperature": 0, "timeout": 120}

# --- response cache ------------------------------------------------------


def prompt_key(model_name: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
    return f"{model_name}:{digest}"


class ResponseCache:
    """Append-only JSONL store of successful model replies.

    Holds the reply text and measured latency, never the prompt or any
    key. Lines that fail to parse (an interrupted write) are ignored.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: Dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._records[record["key"]] = record

    def __len__(self) -> int:
        return len(self._records)

    def get(self, model_name: str, prompt: str) -> Optional[dict]:
        return self._records.get(prompt_key(model_name, prompt))

    def put(self, model_name: str, prompt: str, output: str, latency_ms: float) -> None:
        record = {
            "key": prompt_key(model_name, prompt),
            "model": model_name,
            "output": output,
            "latency_ms": round(latency_ms, 1),
        }
        self._records[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def needed_calls(
    questions: Sequence[Question], models: Sequence[ModelSpec]
) -> List[Tuple[ModelSpec, str]]:
    """Every distinct (model, prompt) a full grid search needs.

    Ordered question by question, cycling through models within each, so
    a partial collection covers whole questions and spreads load across
    providers.
    """
    retriever = Retriever(q.context for q in questions)
    steps = [context_step(retriever, k) for k in CONTEXT_VARIANTS.values()]
    calls = []
    for question in questions:
        prompts = list(dict.fromkeys(step(question.question) for step in steps))
        for prompt in prompts:
            for model in models:
                calls.append((model, prompt))
    return calls
