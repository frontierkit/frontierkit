# Pick the best models for your questions, for free

You have a task, a few candidate models and a few ways to prompt them. Which
combination answers best, and which answers fast enough? This guide runs every
combination on your own questions, scores the answers, and tells you.

It uses only free model endpoints, so it costs nothing but time.

## 1. Install

```bash
pip install "chatcortex[real]"
git clone https://github.com/siddharth1012/chatCortex
cd chatCortex
```

The `real` extra adds the OpenAI SDK, which is used to talk to every vendor
below; none of them is OpenAI.

## 2. Get free keys

Any one is enough; more vendors means more to compare.

| Vendor | Where | Variable |
|---|---|---|
| SAIA (GWDG Chat AI) | Academic Cloud / KISSKI | `SAIA_API_KEY` |
| NVIDIA API catalog | build.nvidia.com | `NVIDIA_API_KEY` |
| Groq | console.groq.com | `GROQ_API_KEY`, plus `GROQ_MODEL` |
| OpenRouter | openrouter.ai | `OPENROUTER_API_KEY` |

Set them as environment variables. On Windows, search the Start menu for
"Edit environment variables for your account" and add them under User
variables, then open a new terminal. On macOS or Linux:

```bash
export SAIA_API_KEY=...
```

Groq renames its models often, so the script has no default for it: set
`GROQ_MODEL`. If you leave it out, the script prints the models your key can
use.

## 3. Write your questions

Make a CSV file with a `question` column, an `answer` column, and optionally a
`context` column. Separate acceptable answers with `|`:

```csv
question,answer,context
Which organ produces insulin?,pancreas,"Insulin is a hormone made by beta cells in the pancreas. ..."
What is the largest ocean on Earth?,Pacific Ocean|Pacific,"The Pacific Ocean is the largest ..."
```

- Use questions like the ones your application will really get. Ten to thirty
  is a good start.
- Keep reference answers short. The default score compares words, so a
  one-phrase reference judges a one-phrase answer fairly.
- Add a `context` column if your application gives the model reading material.
  The script then compares answering with and without it.

`examples/pick_models_questions.csv` has eight general-knowledge questions to
try first.

## 4. Run it

```bash
python examples/pick_models.py --questions my_questions.csv
```

It first checks each model with one call and skips any that fail, printing the
vendor's error and, where it can, the model names your key may use. Then it
tries every combination and prints progress as it goes.

The number of calls is models × prompt styles × context options × questions.
With three models, two prompt styles, context on and off, and 10 questions,
that is 120 calls. Calls are spaced to stay under free-tier limits, and some
free endpoints take seconds or even minutes per answer, so start with
`--max-questions 5`.

## 5. Read the results

Each row is one setup:

| Column | Meaning |
|---|---|
| quality | mean score over your questions, from 0 to 1 |
| answered | share of questions that got any answer |
| latency | mean time per answer |
| `*` | not beaten on both quality and speed by any other setup |

Below the table the script names the setup with the best quality and, if one
exists, a faster setup within 0.05 of it.

The default score is word overlap with your reference (`--scorer f1`). Use
`--scorer contains` when answers may be long but must mention the reference,
or `--scorer exact` for labels and numbers.

With few questions, differences of a few hundredths are noise. Add questions
before relying on a close call.

## 6. Compare your own options

Everything compared is listed near the top of `examples/pick_models.py`:

- `PROMPT_STYLES`: prompt templates, with `{question}` where the question goes.
- `CONTEXT_OPTIONS`: whether the context column is included.
- `MODELS`: one row per model, with its vendor's base URL and the gap kept
  between calls. Any vendor or local server that accepts OpenAI-style chat
  requests works, Ollama included.

Your own code can be a step too. Each step is a plain Python function that
receives the previous step's output: a retriever over your documents, a
re-ranker, a check on the answer. See "Run your own functions" in the README.

## 7. When there are too many combinations

Trying everything is the right choice while it stays affordable. When it
doesn't, let ChatCortex choose what to try. `run_closed_loop` with an eval set
scores a few setups per round, estimates how good each component is from the
scores, and proposes the next setups from those estimates; `explore=N` makes it
also try untried setups each round.

On ChatCortex's own question-answering benchmark this finds the best setup in
about the same number of calls as picking setups at random, so it is not yet a
reason on its own to prefer the loop over trying everything; see
`benchmarks/qa_frontier/`.

## An example result

On 12 questions from the SQuAD reading-comprehension set, with three free models
and three amounts of context, the benchmark in `benchmarks/qa_frontier/`
measured (word-overlap score, mean latency per answer):

| Model | Context | Quality | Latency |
|---|---|---|---|
| Gemma 31B on SAIA | best paragraph | 0.95 | 230 ms |
| Gemma 31B on SAIA | best three paragraphs | 0.95 | 266 ms |
| Gemma 31B on NVIDIA | best paragraph | 0.95 | 14,516 ms |
| Llama 8B on SAIA | best paragraph | 0.33 | 250 ms |
| Gemma 31B on SAIA | none | 0.10 | 259 ms |
| Llama 8B on SAIA | none | 0.00 | 247 ms |

Three things stand out: without reading material even the strong model gets
almost nothing right; the small model is no faster here and much worse; and
the same model was over 60 times faster on one vendor than on the other. Your
questions and vendors will give different numbers, which is the reason to
measure.
