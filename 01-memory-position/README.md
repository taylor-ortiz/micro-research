# 01 · Memory position: where a memory sits decides whether it gets used

Research brief, Nov 2026. Code and data are in this folder; see [Reproduce](#reproduce) and [Files](#files) at the end.

## Summary

**On Claude Haiku 5.5, a stored preference listed first in a long memory list was followed almost every time, and one listed in the middle almost never.** With about 1,450 memories loaded, a preference at the top was followed 98–99% of the time and 0% of the time from the middle. Claude Sonnet 5.5 and Claude Opus 5.5 followed it 100% of the time at the top, middle and end.

**Research question:** when an agent carries a list of stored memories, does it follow a given preference less often depending on where that preference sits in the list, and does that depend on how big the list is and which model reads it?

This is the "Lost in the Middle" effect ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)), applied to an agent's procedural memory. On Haiku it starts as a U-curve with small stores and collapses into a cliff as the store grows, until only the top of the list survives.

![Results grid](posts/images/results-grid.png)

## Background

Agents increasingly carry long-term memory: preferences, facts and rules saved from earlier conversations and put back into the prompt. The simplest design injects the whole list. Long-context research has shown that models use information at the start and end of a prompt better than information in the middle. If that holds for memory, a rule a user stated once could be silently ignored depending on where it happens to land in the list.

## Working assumptions

Credences were not recorded before the runs. Future studies should record them here, before running. The beliefs going in were:

- Adherence follows a U-curve over position, as in Lost in the Middle.
- The middle gets worse as the memory list gets longer.
- A preference that changes the whole reply (a table) is harder to apply than a small one (a sign-off word).
- The effect would show on every model, shrinking but not vanishing on larger ones.

## Theory

A model reading a long, flat list of similar memories gives more weight to the edges of the list than to its middle. The more items there are, the weaker the middle becomes, and eventually the end weakens too.

Falsifiable predictions:

- A preference at 25–75% depth is followed at least 30 points less often than one at 0% or 100%.
- The drop grows with the size of the memory list.
- A preference that is easy to apply survives in the middle longer than one that reshapes the whole reply, if the failure is in applying rather than noticing.

## Subject and contrast

**Subject: Claude Haiku 5.5**, answering 100 neutral questions for one synthetic user whose memory list holds about 45, 360 or 1,450 preferences, with one target preference (the needle) placed at a set position. Haiku is cheap enough to run the full grid: 3 sizes, 5 positions and a no-needle baseline.

**Contrast: Claude Sonnet 5.5 and Claude Opus 5.5**, on 30 of the same questions with the marker needle at the top, middle and end of the two larger lists. If the effect is a general property of long memory, they should show it too. They did not: both followed the needle in every reply.

## What counts as proof

Every hit must trace to the needle. A reply counts as following the memory only if it does something the model does not do on its own.

| Term | Meaning | How we know |
|---|---|---|
| Present | The needle is in the prompt at the intended position | Asserted in every trial, with its token offset logged |
| Applied | The reply follows the needle | Deterministic check: a Markdown table, or a reply ending in "Cheers" |
| Caused by the memory | The reply would not do this without the needle | No-needle baseline: 0% tables at every size; 0 of 300 replies end in "Cheers" |
| Not caused by neighbors | No other memory pushes toward the same behavior | Haystack screened for format words, similarity to the needle and anything that could change a reply's structure |

**Two needles.** Tables can appear out of model habit, so a second needle with a near-zero natural rate ("end with the single closing word Cheers") was added. Every Cheers hit proves the memory was used. Comparing the two also separates noticing a memory from applying it.

**Not claimed:** anything about what happens inside the model.

## Setups

Every trial gets the same prompt template and question set. Only the needle's position, the list size and the answering model change.

| Factor | Levels |
|---|---|
| Needle position (share of the list, by item index) | 0% (top), 25%, 50% (middle), 75%, 100% (end) |
| Memory list size | 2k, 16k, 64k tokens: about 45, 360 and 1,450 items at ~44 tokens per serialized item |
| Needle | Tables, Cheers |
| Model | Haiku 5.5 (full grid); Sonnet 5.5 and Opus 5.5 (Cheers, 16k and 64k, top/middle/end) |
| Baseline | No needle, same haystack (tables on Haiku) |

| Needle | Text | Tokens |
|---|---|---|
| Tables | User prefers all responses to be formatted as Markdown tables. | 20 |
| Cheers (marker) | User prefers every reply to end with the single closing word Cheers. | 20 |

## Test material

**Haystack:** `claude-opus-5-5` generated 4,440 one-sentence user preferences across 15 categories: 12 everyday-life categories (food, travel, pets, hobbies…) and 3 "format-neutral" ones (currency and dates, naming, choice of examples). 3,352 were kept (75%) and frozen as `data/haystack_v1.json`.

| Check | Result |
|---|---|
| Candidates generated | 4,440 |
| Kept | 3,352 (75%) |
| Dropped: token length | 1,002 |
| Dropped: banned word | 52 |
| Dropped: near-duplicate | 19 |
| Dropped: exact duplicate | 15 |
| Dropped: LLM review | 0 |
| Reviewer canaries caught | 102 / 102 |
| Token length of kept items | 18–24, mean 20.8 (needle 20) |
| Format-neutral share | 16.8% |
| Max similarity to needle | 0.03 (limit 0.40) |
| Smallest category | units/locale, 171 items |

| Category | Preference | Kind |
|---|---|---|
| Food | User prefers oat milk over dairy milk in coffee and tea drinks. | Inert |
| Travel | User usually books aisle seats on flights longer than three hours. | Inert |
| Address | User prefers being addressed by their first name in every conversation. | Output, format-neutral |
| Currency | User prefers prices in euros for European travel budgets. | Output, format-neutral |

The mix is deliberate: about 85% are facts about the user's life, and about 15% change a reply without touching its structure. With those in the list, the needle is not the only item that tells the agent how to respond, so it does not stand out just by being an instruction.

**Questions:** 114 drafted across 12 topics. 6 were removed for overlapping specific haystack memories and 11 for producing a table with no memory at all (9 of those only on Opus, 2 on both models). 100 remain, with 3 more surviving questions left out to balance topics: `queries/queries_v1.csv`.

## Controls and confounds

- **Held fixed:** the prompt template, the message slot memory goes in (the system prompt), effort and sampling settings.
- **Length matching:** every item is 18–24 tokens on the target tokenizer (the needle is 20). This text measured about 2 tokens per word on `claude-opus-5-5`, and Haiku 5.5 counted identically on 300 of 300 sampled items, so length matching holds on both.
- **Leakage:** the model sees only `id` and `preference`. IDs are random 4-character hex codes, so they don't reveal list position. Category and token count stay in a sidecar file.
- **Neighbors:** each memory size uses 5 fixed haystack orders, with questions split evenly across them. A given order is identical across all positions and the baseline, so position is the only thing that changes between paired trials.
- **No retriever:** the ordered list is injected directly, so no ranking step can move the needle.
- **Contamination:** banned words (markdown, table, list, format, metric, question, …), TF-IDF similarity to the needle below 0.40, and an LLM reviewer that flagged anything that could change a reply's structure, length, tone, units, spelling or how it opens or ends. Three planted canaries in every review batch of 100 checked the reviewer.
- **Model habit:** questions that drew a table with no memory were dropped, and the marker needle has no natural rate.
- **Known confound:** the needle is one of the few instructions among mostly factual memories. An all-procedural haystack would test position alone; it was planned for stage 2 and not run.

## Measurements and analysis

| Measure | What it is |
|---|---|
| Adherence (primary) | Share of replies that follow the needle, per size and position |
| Baseline rate | The same, with no needle in the list |
| Opening strictness | Share of table replies that open with the table |
| Cost | Tokens and dollars per run, from the API's reported usage |

**Grading** is deterministic. A table is a Markdown header row followed by a `|---|` separator row. "Cheers" means the last word of the reply is "Cheers."

**Analysis.** Each cell reports a 95% interval that treats replies as independent. The main contrast is mean adherence at 25–75% depth against the 0% and 100% ends, at each size.

## Hypotheses against outcomes

Hypotheses and thresholds were written before the runs. The 30-point threshold is in absolute percentage points.

| Hypothesis | Supported if | Outcome |
|---|---|---|
| H1: adherence follows a U-curve over depth | 25–75% depth at least 30 points below 0% and 100% | **Supported against the top at every size; against the end at 2k and 16k only.** At 64k the end collapses too (tables 7%) |
| H2: the mid-list drop grows with list size | Drop small at 2k, largest at 64k | **Supported.** Tables at the end fall 96 → 46 → 7 as the list grows |
| H3: a competing format rule lowers adherence, and the later one wins | Adherence depends on order relative to the competing rule | **Not tested** (stage 2) |
| H4: an agent framework shows a larger effect than a raw call | Path × depth interaction | **Not tested** (stage 3) |

A dry run (Haiku, 64k, 10 questions, 2 shuffles) gave 100% tables at the top, 0% at the middle and end, and 0% in the baseline, for $0.10. That was a pilot signal, not a result.

## Results

On Haiku, a preference at the top of memory was followed almost every time, and it faded fast everywhere else as memory grew. Sonnet and Opus showed no position effect.

**Tables needle (Haiku 5.5): share of replies containing a Markdown table.** 100 questions per cell, 95% intervals. Without the needle, 0% of replies contained a table at every size.

| Memories | Top (0%) | 25% | Middle (50%) | 75% | End (100%) |
|---|---|---|---|---|---|
| ~45 (2k tokens) | 100 [96–100] | 80 [71–87] | 46 [37–56] | 42 [33–52] | 96 [90–98] |
| ~360 (16k tokens) | 100 [96–100] | 16 [10–24] | 2 [1–7] | 2 [1–7] | 46 [37–56] |
| ~1,450 (64k tokens) | 99 [95–100] | 1 [0–5] | 0 [0–4] | 0 [0–4] | 7 [3–14] |

**Cheers needle: share of replies ending with "Cheers."** Haiku: 50 questions per cell. Sonnet and Opus: 30 per cell, top/middle/end only. A 30/30 cell has a 95% interval of roughly 89–100%.

| Model | Memories | Top | 25% | Middle | 75% | End |
|---|---|---|---|---|---|---|
| Haiku 5.5 | ~45 | 100 | 100 | 100 | 92 | 100 |
| Haiku 5.5 | ~360 | 100 | 48 | 2 | 4 | 80 |
| Haiku 5.5 | ~1,450 | 98 | 0 | 0 | 0 | 34 |
| Sonnet 5.5 | ~360 | 100 | — | 100 | — | 100 |
| Sonnet 5.5 | ~1,450 | 100 | — | 100 | — | 100 |
| Opus 5.5 | ~360 | 100 | — | 100 | — | 100 |
| Opus 5.5 | ~1,450 | 100 | — | 100 | — | 100 |

**Noticing vs. applying.** With ~45 memories, Haiku followed the Cheers needle about 100% of the time from every position but made a table only about 45% of the time from the middle. So it saw mid-list memories but didn't always apply a preference that changes the whole reply. With larger lists both needles failed, which points to the memory not being noticed at all.

**Opening strictness.** At 2k, table replies opened with the table 84% of the time when the needle was first, against 29–42% at other depths.

**Shuffles.** The same cell varied widely across the 5 haystack orders: tables at 2k/middle ranged 20–95%, and Cheers at 16k/25% ranged 10–100%.

## Updates

- **Position is a real failure mode for small models.** On Haiku, the same preference went from always followed to never followed by moving it from the top of the list to the middle.
- **The U-curve is only the start.** As the list grows, the end of the list stops helping, and only the top survives.
- **Two kinds of failure.** In short lists Haiku notices mid-list memories but applies heavy ones less often. In long lists it misses them entirely.
- **Larger models read long memory reliably,** at least up to ~1,450 items and for an easy-to-apply preference.
- **The open question moved from reading to selection.** If a long list is unreliable on small models, the next question is which memories should go into the prompt at all. That became [study 02](../02-memory-routing/).

## Construct validity

Adherence measures memory use because each question was built so the needle's behavior does not happen by itself:

- Questions that produced a table with no memory were dropped, and the no-needle baseline stayed at 0% tables at every size.
- The marker needle has a base rate of 0 of 300 replies on Haiku, so every hit is memory-driven.
- The needle was asserted present at its intended position in every prompt, with nothing trimmed.
- The reviewer caught 102 of 102 planted canaries, so items that could change a reply's shape were filtered out.

The main threats are the intervals and the needles. The intervals treat replies as independent, but shuffles matter a lot, so they are too narrow; a full analysis should cluster by shuffle. And only two preferences were tested: Cheers is the easiest to apply, so the larger models could still apply a heavier preference less often by position.

## Novelty and related work

The effect itself is known: [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) for retrieved documents, and needle-in-a-haystack tests for long-context recall.

What's new here:

- The effect measured on an agent's memory list, with a length-matched, screened haystack of user preferences rather than documents.
- A marker needle with no natural rate, which separates noticing a memory from applying it.
- A size sweep showing the U-curve collapsing into a top-only cliff on a small current model, and no effect on larger current models at the same sizes.

## Cost and feasibility

The experiment runs cost about **$20** in Claude calls. Haystack generation, screening and pilot runs aren't included.

| Plan | Model | Needle | Sizes | Positions | Questions | Calls | Cost |
|---|---|---|---|---|---|---|---|
| `haiku_tables` | Haiku 5.5 | Tables | 2k, 16k, 64k | 5 + no needle | 100 | 1,800 | $1.52 |
| `haiku_marker` | Haiku 5.5 | Cheers | 2k, 16k, 64k | 5 | 50 | 750 | $0.74 |
| `sonnet_marker` | Sonnet 5.5 | Cheers | 16k, 64k | 3 | 30 | 180 | $5.50 |
| `opus_marker` | Opus 5.5 | Cheers | 16k, 64k | 3 | 30 | 180 | $11.87 |

Costs are computed from the token usage the API returned. Everything ran on 2026-11-23 with no errors or truncated replies. Models: Haiku 5.5, Sonnet 5.5 and Opus 5.5 answer; Opus 5.5 generated the haystack.

## Ethics

The user is synthetic, so no real personal data was used, and the haystack excludes personal data by rule. The finding matters for real systems: a user who states a rule once can have it silently ignored depending on where it lands in memory. No keys or credentials are stored in the repository.

## Limits and next steps

**Limits:** two preferences, one prompt template, memories in the system prompt, synthetic memories, up to ~1,450 items. Only procedural preferences were tested, not facts or past conversations. Sonnet and Opus were tested only with the Cheers needle, at 3 positions and 30 questions per cell, with no no-needle baseline. Intervals are too narrow because shuffles are not clustered.

**Next checks:**

1. A small tables-needle check on Sonnet, to test whether a heavier preference is applied less often by position on larger models.
2. Re-analysis clustered by shuffle.
3. Stage 2: query relevance tiers, a competing format rule (H3), an all-procedural haystack, and the five-needle family (tables, follow-up question, summary first, metric units, British spelling), which is defined in `haystack/config.py` and already screened against.
4. Stage 3: the same conditions through an agent framework (H4).

**Moved on:** which memories go into the prompt at all became [study 02](../02-memory-routing/). Ordering policies (most recent first, by importance, rules at both ends) are planned for study 03.

## Appendix A: Procedure

This appendix is detailed enough for a coding agent to rebuild the study.

1. **Generate** (`haystack/generate.py`): Opus writes about 4,400 one-sentence preferences in the needle's voice ("User prefers…", "User likes…", "User usually…"), across 15 categories with roughly equal counts.
2. **Validate and freeze** (`haystack/validate.py`): apply the filters in Appendix B, run the LLM reviewer with 3 canaries per batch of 100, and freeze `data/haystack_v1.json` with its SHA-256 in `data/haystack_v1.meta.json`.
3. **Screen questions** (`harness/screen.py`): run each drafted question with no memory and drop any that produce a table.
4. **Build a trial** (`harness/run.py`): take the size's fixed shuffle, cut it to the target token count, insert the needle at the depth's item index, assert it is present, and log its token offset.
5. **Prompt**, identical in every trial:

```text
System: You are a helpful assistant. Below are memories you have stored about this user from earlier conversations, as JSON. Use them when they are relevant.

<memories>
[{"id": "a3f9", "preference": "User prefers oat milk over dairy milk in coffee and tea drinks."}, ...]
</memories>

User: <question>
```

Answer settings: effort medium, `max_tokens` 8000.

6. **Grade** deterministically and log every trial, with settings, token usage, grade and the full reply, to `data/runs/<plan>.jsonl`.

## Appendix B: Decision rules

- **Haystack item:** one sentence, one preference, 18–24 tokens on the target tokenizer, no banned word, no negations or conditionals, no personal data.
- **Duplicates:** drop exact duplicates and pairs with TF-IDF cosine ≥ 0.80.
- **Needle similarity:** drop any item with TF-IDF cosine ≥ 0.40 to the needle.
- **Reviewer:** drop any item flagged as able to change a reply's structure, length, tone, units, spelling or how it opens or ends. The reviewer must catch every planted canary.
- **Question inclusion:** drop questions that overlap specific haystack memories or produce a table with no memory.
- **H1 threshold:** 30 absolute percentage points.
- **Cost cap:** a run refuses to start if its estimated cost exceeds `--max-cost` (default $5).

## Appendix C: Recipe, placing memories on small models

What this study supports, for an agent that injects a memory list:

1. **On small models, keep the list short or put what matters first.** Standing rules a user stated once belong at the top of the memory block, not wherever they happened to be saved.
2. **Don't rely on the end of the list.** It helps in short lists and stops helping as the list grows.
3. **Test your own position effect** with a marker preference that has no natural rate (a fixed sign-off word), placed at the top, middle and end of your real memory block.
4. **Larger models are more forgiving** at these sizes, but a heavier preference is untested on them.

Study 02 found a stronger fix: choose a short list of relevant memories per question instead of injecting the whole store.

## Reproduce

Credentials come from the SDK's normal lookup (for example an `ant auth login` profile). No keys are read from or stored in this repository.

```bash
uv sync
ant auth login            # or copy .env.example to .env and add an API key
```

```bash
uv run python -m haystack.generate      # ~4,400 candidates
uv run python -m haystack.validate      # filters, then freezes haystack_v1
uv run python -m harness.screen         # screens questions with no memory
uv run python -m harness.run haiku_tables --estimate
uv run python -m harness.run haiku_tables
```

Every plan is defined in `harness/run.py`. Runs are resumable and refuse to start if the estimated cost exceeds `--max-cost` (default $5).

## Files

| Path | Contents |
|---|---|
| `haystack/` | Generation, validation and settings (models, needles, banned words, thresholds) |
| `harness/` | Smoke test, question screen and the main runner |
| `queries/` | Drafted questions and the final 100 |
| `data/haystack_v1.json` | The frozen haystack, exactly what the model saw |
| `data/haystack_v1.meta.json` | Hash, needle and per-item category, kind and token count |
| `data/candidates.jsonl`, `data/drops.jsonl` | Every generated item, and every dropped item with its reason |
| `data/validation_report.json` | Haystack statistics and reviewer canary results |
| `data/screen_results.json` | No-memory screen of the 114 drafted questions |
| `data/runs/*.jsonl` | Every trial: settings, token usage, grades and the full reply |
| `posts/` | Images and the writing template for the accompanying post |
