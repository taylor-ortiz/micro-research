# 02 · Memory routing: a relevance judge beats similarity search

Research brief, Nov 2026. Code and data are in this folder; see [Reproduce](#reproduce) and [Files](#files) at the end.

## Summary

**Having a fast classifier pick the 9 or so memories relevant to each question worked as well as putting all 1,000 memories in the prompt, while sending the answering model 98% fewer tokens.** Jev, a fast classifier, scored the groups and then the memories in them. Answers then scored 93% on Haiku 5.5 and 94% on Sonnet 5.5, against 87% and 96% for injecting every memory. Similarity search, the usual way agents fetch memories, scored 60–63%.

**Research question:** how should an agent choose which memories go into the prompt, so the right one gets used at the lowest cost? Every setup ends by injecting memories. They differ only in how they pick them.

The bottleneck turned out to be retrieval, not reading. Both models used a memory almost every time it reached the prompt. What failed was similarity search, on memories that matter without sharing words with the question: standing rules such as "use metric units" and constraints such as "allergic to peanuts."

## Background

[Study 01](../01-memory-position/) showed that where a memory sits in the prompt can decide whether it gets used. With about 1,450 memories loaded, Haiku 5.5 followed a preference listed first 98–99% of the time, and one listed in the middle 0% of the time. Sonnet and Opus showed no position effect.

That left the next question: if a long memory list is unreliable, how should an agent decide which memories go into the prompt? The starting intuition was that memory types need different handling. Facts and past sessions can be searched. Standing rules apply to every reply but look like none of them, so search should miss them.

## Working assumptions

Credences were not recorded before the runs. Future studies should record them here, before running. The beliefs going in were:

- Similarity search misses standing rules much more often than facts.
- Pinning every rule into every prompt fixes rule adherence.
- Grouping memories into described topics, and letting a model decide which groups to search, beats searching all memories at once.
- Richer group descriptions (what a group covers, when it applies, examples) help the model choose.
- Injecting every memory is the accuracy ceiling, but costs too much to be the answer.
- On Sonnet, failures come from finding memories, not using them. Haiku also loses memories in long prompts.

## Theory

For agent memory, retrieval is the bottleneck, not reading. What matters for a request is often related to it without resembling it. "Use metric units" matters for "How far is a half marathon?", and "allergic to peanuts" matters for "Plan a Thai takeout order." Neither shares wording with its question, so similarity search ranks them low.

Falsifiable predictions:

- Similarity search misses style rules at least twice as often as facts.
- Organizing memories (searching groups separately, pinning rules) recovers part of the gap.
- Replacing similarity with a judge of relevance, one that decides whether each memory would change the reply, closes most of the gap to injecting every memory.

## Subject and contrast

**Subject: Claude Haiku 5.5**, answering for one synthetic user with a store of 1,000 memories, plus one gold memory per question. Haiku is where retrieval design matters most: study 01 showed it loses memories in long prompts, so injecting everything is a weak option. It is also cheap enough to run every setup at three positions on 188 held-out questions.

**Contrast: Claude Sonnet 5.5**, on 99 of the same questions with the gold memory in the middle. Sonnet reads long prompts well, so injecting everything should look much better on Sonnet. If retrieval is the bottleneck, the setups that search should score about the same on both models. That is what happened: injecting everything scored 96% on Sonnet and 87% on Haiku, while every search-based setup landed within about six points across the two.

## What counts as proof

Every pass must trace to a known right memory. Each question is built around one **gold memory**, the memory its pass check asserts against, and four facts are recorded per trial.

| Term | Meaning | How we know |
|---|---|---|
| Reached | The gold memory is in the prompt | Exact, from the trial log |
| Applied | The reply passes its pre-written check | Tool-call order, regex, word list plus judge, or judge |
| Caused by the memory | The reply changes when the memory changes | No-memory baseline and the swap test |
| Applied correctly | No distractor applied instead | Distractor screen; over-application on no-memory questions |

**Swap test.** Each gold memory has a minimal twin (metric ↔ imperial, peanuts ↔ shellfish). A reply that follows whichever twin is present shows the memory caused it.

**Validation.** A question was kept only if it fails with no memory, and follows its gold memory, and separately its twin, in 4 of 4 tries when given only that memory.

**Secondary golds.** Every gold memory was checked against all 1,000 store memories. A store memory that would pass the gold check on its own became a secondary gold for that question. One that contradicted the gold, or supported its twin, was replaced. Retrieval is reported both for the planted gold and for any gold.

**Not claimed:** anything about what happens inside the model. "Noticed" is inferred only from the reasoning summary.

## Setups

Every setup gets the same user message, prompt wording, store and grading. Only the way memories are picked changes.

| Setup | Role | How memories are picked |
|---|---|---|
| A. Inject everything | Control | All 1,000, every time |
| E. Search everything | Control | Shared similarity search over all memories, top 10 |
| B1. Rules always injected | Organizing | Every procedural memory, plus search over the rest |
| G. Search every group | Organizing | Search inside each of 26 topic groups, top 10 each |
| C1–C3. Ontology reading | Organizing | Haiku reads a group index (C1 names; C2 + descriptions; C3 + "applies when" and examples), picks groups, and search runs inside them |
| C-types | Organizing | The same, over 4 type groups |
| C-embed | Organizing | The 3 groups most similar to the message are searched |
| C-Jev | Organizing | Jev scores every group; groups at 0.1 or above are searched |
| **C-Jev+** | Relevance judge | Jev picks groups (0.1), then scores every memory in them; those at 0.7 or above go in, at most 20 |
| L-cat | Relevance judge | Haiku scores the same candidates as C-Jev+; those at 0.6 or above go in |
| L-all | Relevance judge | Haiku scores all 1,000; those at 0.6 or above go in |
| D. Perfect retrieval | Ceiling | Only the gold memory |
| No memory | Baseline | Nothing |

**Shared search.** BM25 and OpenAI `text-embedding-3-small`, with rankings merged by reciprocal rank fusion, top 10, using the user's message as the query. No LLM takes part.

**Ontology.** 26 topic groups under 4 types (style rules, task procedures, facts, past sessions), proposed by Opus from the store without seeing questions. Every memory, gold included, was filed by Haiku. Each entry has a name, a description, "applies when", 3 examples, a size and type-specific extras.

**Jev** is TypeSafe's decision model: it answers yes/no questions about a text with a calibrated probability, many questions per call.

## Test material

**Store:** 1,000 distractors in 5 fixed shuffles.

- 600 facts from study 01's `haystack_v1`
- 250 generated past sessions ("On 2026-MM-DD we ...")
- 150 procedural memories: 36 task decoys naming mock tools, 40 every-reply style rules and 74 content-choice rules

**Questions:** 263, each with a gold memory, a swap twin and a pass check written before the run.

| Scenario | Questions | Example | Graded by |
|---|---|---|---|
| Task procedure | 45 | "Get me a hotel near the convention center in Austin" → "search `web_search` first" | Tool order |
| Style rule | 17 | "How far is a half marathon?" → "use metric units" | Regex, first match wins |
| Fact | 32 | "Suggest a weeknight dinner" → "vegetarian" | Judge |
| Allergy | 51 | "Plan a Thai takeout order" → "allergic to peanuts" | Word list, judge on hits |
| Past session | 38 | "Which section of the grant proposal still needs writing?" → "Sept 17: evaluation plan left" | Judge |
| Unclear type | 40 | "Plan a decluttering month" → "one room per weekend, never weekdays" | Judge |
| No memory | 40 | "How do black holes form?" → none | Judge (over-application) |

**Split:** 75 development questions for tuning; all results are on the other 188.

## Controls and confounds

- **Held fixed:** the user message, which is also the search query; prompt wording; store and shuffles; search method; grading.
- **Position:** the gold memory is placed at a set depth (top, middle or end) of whatever list the model reads. Search rank never sets order, and ties break by an ID hash.
- **Leakage:** memory IDs are opaque and change per shuffle. Gold memories match distractors in length and voice.
- **Contamination:** an exhaustive screen of every gold against all 1,000 memories. Haiku flagged 1,431 candidates and Sonnet confirmed 136. 86 contradicting memories were replaced, and 3 secondary golds recorded.
- **Tuning:** the three cutoffs (Jev groups 0.1, Jev memories 0.7, LLM picker 0.6) were chosen on development questions only, by one rule written in advance.
- **Defaults that would favor one setup:** where Haiku's own default already matched the generated gold, gold and twin were swapped, so a pass always requires the memory.

## Measurements and analysis

| Measure | What it is |
|---|---|
| Adherence (primary) | Share of gold-question trials whose reply passes its check |
| Gold in prompt | Planted gold, and any gold, present in what the model read |
| Read failure | Gold in the prompt, but the reply fails |
| Reasoning critic | Haiku labels the reasoning summary: not mentioned, set aside, or applied |
| Over-application | No-memory questions where a memory was forced into the answer |
| Cost | Tokens and dollars for picking and for answering, per question |

**Analysis.** 95% intervals come from a bootstrap over questions (2,000 draws), so one question's three depths are not treated as independent. Setups are compared by paired differences on identical question-and-depth trials. Haiku and Sonnet are compared on the same 99 questions at middle depth.

## Hypotheses against outcomes

Hypotheses and thresholds were written before the main run. Results are Haiku, on the 188 held-out questions.

| Hypothesis | Supported if | Outcome |
|---|---|---|
| H1: reading the ontology (C3) works as well as injecting everything, more cheaply | Within 5 points of A, at most 25% of A's tokens, at least 10 points above search | **Not supported:** C3 61% vs. A 87%; no better than search |
| H2: search misses style rules more than facts | Style rules missed at least 2× as often as facts | **Direction only:** 64% vs. 38% missed (1.7×). Task procedures were missed least (17%) |
| H3: choosing groups per question helps on unclear-type memories | C beats B1 on unclear-type questions | **Mostly not:** C1–C3 57–59 of 81 vs. B1 57; C-types 64 |
| H4: injecting everything breaks first as the store grows | A falls below B1 at 5,000 memories | **Not yet tested** (stage 4) |
| H7: position matters for injecting everything on Haiku | Middle at least 30 points below top | **Direction only:** 97% top vs. 81% middle |
| H8: richer ontology entries help | C1 < C2 < C3 | **Not supported:** 62%, 65%, 61% |
| H9: grouping alone helps | At least 10 points above search on style rules and tasks | **Supported:** +52 on style rules (88% vs. 36%), +15 on tasks |

C-Jev+ and the LLM pickers were added after the main run, to test the theory's third prediction directly. Their cutoffs were tuned on the development questions before they ran on the held-out ones.

## Results

Jev picking memories (C-Jev+) beat injecting everything on Haiku and tied it on Sonnet. It used about 9 memories instead of 1,000.

| Setup | Haiku (95% CI) | Sonnet (95% CI) | Answer tokens per question |
|---|---|---|---|
| D. Perfect retrieval | 98% (95–100) | 93% (88–97) | 0.9k |
| **C-Jev+, Jev picks memories** | **93% (89–96)** | **94% (89–98)** | **1.3k** |
| L-all, Haiku picks from all | 92% (88–96) | 87% (80–93) | 1.3k |
| L-cat, Haiku picks within groups | 91% (86–94) | 89% (82–95) | 1.3k |
| A. Inject everything | 87% (83–91) | 96% (92–99) | 66k |
| G. Search every group | 74% (67–80) | 74% (65–82) | 18k |
| C-Jev, Jev picks groups | 71% (64–78) | 75% (66–83) | 10k |
| B1. Rules always injected | 70% (63–77) | 73% (64–81) | 11k |
| C-types | 67% (59–74) | 66% (57–75) | 6k |
| C2. Ontology + descriptions | 65% (58–71) | 64% (54–74) | 10k |
| E. Search everything | 63% (55–70) | 60% (49–69) | 1.5k |
| C1 / C3 | 62% / 61% | not run | 7k / 24k |
| C-embed | 44% (37–52) | not run | 3k |
| No memory | 7% | 4% | 0.7k |

**Paired on the same questions:**

- C-Jev+ vs. injecting everything: Haiku **+6 points (+1 to +10)**, Sonnet **−2 (−8 to +4)**, a tie.
- C-Jev+ vs. the LLM pickers: a tie on Haiku (−1 to −2). On Sonnet the LLM pickers trail by 5–7 points, significantly for L-all (−7, −14 to −1).
- C-Jev+ vs. the best similarity-based setup (G): about +19 to +20 on both models.

**By scenario (Haiku, C-Jev+):** allergy 102/102, past sessions 75/75, style rules 31/33, task procedures 79/90, unclear type 72/81, facts 53/63. Every similarity-based setup scored 18–28 of 102 on allergy.

**Position (Haiku):** C-Jev+ scored 94% / 93% / 92% at top, middle and end. Injecting everything scored 97% / 81% / 84%.

**Swap test:** when the twin reached the prompt, replies followed it 92% of the time across the main setups and 96% for C-Jev+.

**Cost per question** (standard rates, picking plus answering):

| Setup | Haiku answering | Sonnet answering | Picking time |
|---|---|---|---|
| A. Inject everything | $0.0071 | $0.140 | — |
| C-Jev+ | **$0.0017** | **$0.013** | ~0.3–1 s |
| L-cat | $0.0035 | $0.015 | ~3.3 s |
| L-all | $0.0057 | $0.017 | ~2.9 s |

Jev reads about 26k tokens per question at $0.042 per million (output free), about $0.001. With Sonnet answering, most of the remaining cost is Sonnet's own reply.

## Updates

- **The answering model is rarely the problem.** Once the gold memory reached the prompt, both models used it almost every time. Whether it got there decided the result.
- **Organizing memories helps, but only partly.** Searching every group, pinning rules and Jev-chosen groups beat plain search by 7–15 points, but stayed 14–17 points below injecting everything on Haiku.
- **Pinning fixes rules specifically.** Pinned rules scored 33 of 33 style rules on Haiku, against 12 for plain search.
- **Reasoning over a richer index did not help.** The ontology ladder was flat, and LLM group choice did no better than searching every group.
- **A relevance judge picking a shortlist is what works.** About 9 memories chosen by relevance, not similarity, matched or beat all 1,000.
- **The judge can be a classifier.** Jev was as accurate as an LLM picker or better, about ten times faster, and cheaper per question.

## Construct validity

Adherence measures memory use because each question was built to depend on its memory:

- Questions passed with no memory were dropped, and each kept question followed its gold, and separately its twin, in 4 of 4 tries when given only that memory.
- The swap test showed replies follow whichever version reaches the prompt: 92% across the main setups, 96% for C-Jev+.
- 48 of 50 sampled grades held up on review, and the Haiku judge agreed with Sonnet on 56 of 60. Allergy replies that Haiku failed were re-checked by Sonnet, since Haiku's errors there were false failures.

The main threat is the store itself: synthetic, screened for conflicts and fixed at 1,000 memories. Real stores hold more borderline-relevant memories.

## Novelty and related work

The parts are familiar: retrieval-augmented generation, rerankers in search, the "lost in the middle" effect ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)), and agent memory systems that search or summarize.

What's new here:

- A controlled comparison of memory selection policies for agents, with the gold memory's position held fixed and a swap test for cause.
- Evidence that similarity search is the bottleneck, specifically for standing rules and memories that need a step of reasoning to connect.
- A fast classifier as the relevance judge, matching injecting every memory with about 98% fewer answer tokens, and matching or beating an LLM picker at about ten times the speed.

Not yet compared: off-the-shelf rerankers such as Voyage or Cohere.

## Cost and feasibility

The study cost about **$37** in Claude calls, across about 9,900 answer trials, mostly through message batches. Jev and the embeddings came to well under $1.

| Part | Trials | Cost |
|---|---|---|
| Build: generation, screens, validation, ontology | — | ~$10 |
| Pilot | 650 | $1.69 |
| Haiku main run and swap test | 6,212 | $6.25 |
| Sonnet main run | 1,051 | $15.17 |
| C-Jev+ (both models, swap test) | 751 | ~$0.75 |
| LLM pickers (both models), including picking | 1,206 | ~$3.80 |

It took about two days, including design. Each run took one to two hours, mostly batch turnaround. Models: Haiku 5.5 and Sonnet 5.5 answer; Opus 5.5 generated the test material; Jev decides relevance; OpenAI `text-embedding-3-small` powers search.

## Ethics

The user is synthetic, so no real personal data was used. Some memories are health-like (allergies, injuries, diabetes), a reminder that real memory systems hold sensitive facts. A selector that reliably surfaces the right memory will also surface sensitive ones, so what is stored, and who can query it, matters as much as retrieval quality. No keys or credentials are stored in the repository.

## Limits and next steps

**Limits:** one synthetic store of 1,000 memories, screened so distractors don't give answers away. Cutoffs were tuned on 75 development questions. Jev's work grows with group size, and no off-the-shelf reranker was compared.

**Next checks, before calling the result general:**

1. **Near-miss memories** (~$5): add 3–5 related-but-irrelevant memories per question ("loves Thai iced tea" alongside "allergic to peanuts") and see whether Jev still ranks the gold first.
2. **Scale** (~$25–30): 5,000 memories, for C-Jev+, plain search, search every group and C-Jev on both models. Injecting everything runs on Haiku only, as a reference. This also tests H4.
3. **A reranker baseline**, if a Voyage or Cohere key is available.

**Moved to study 03:** ordering policies (most recent first, by importance, rules at both ends), including rules first vs. last.

**Ideas this opened:** a growing ontology that splits groups as they fill, so the judge walks a tree and its work stays bounded; proving memory use; memory capture; learned procedures. They're tracked in a separate ideas log.

## Appendix A: Procedure

This appendix is detailed enough for a coding agent to rebuild the study. Code and data are in this folder.

1. **Build** (`build/`): generate items with Opus; run the no-memory screen (drop the question, or swap gold and twin); validate each question with only its gold and only its twin (4 of 4); screen every gold against all 1,000 memories; freeze everything with SHA-256 hashes (`data/frozen.meta.json`).
2. **Prepare a trial** (`build/trials.py`): the corpus is the store in the shuffle's order plus the gold memory, with opaque IDs. Apply the setup's selection policy, place the gold at the trial's depth, and keep the other memories in shuffle order.
3. **Prompt**, identical in every setup:

```text
System: You are a helpful assistant.

Below are memories you have stored about this user from earlier conversations, as JSON. Use them when they are relevant.

[{"id": "m3f2a", "memory": "..."}, ...]

User: Plan a Thai takeout order for two.
```

Task questions add: "You can act with the tools provided. Take action now: make reasonable assumptions about missing details instead of asking the user first." They get up to 3 tool rounds with stub results. Answer settings: effort medium, adaptive thinking with summaries, `max_tokens` 8000.

**4. Jev questions.** State: `{"user_message": <message>}`.

- Group: "Could any memory in this group change or improve the assistant's reply to user_message? Groups that apply to every reply always count. Group entry: {name}. {description}"
- Memory: "Could this memory change or improve the assistant's reply to user_message? Memory: {memory}", up to 250 per call.

5. **LLM picker** (L-cat, L-all): Haiku at low effort reads 100 memories per call and returns the numbers of relevant ones with a score from 0 to 1; unlisted memories count as 0.
6. **Grade** (`build/grade.py`) and **critique** each reply, logging to `data/runs/<plan>.jsonl`.
7. **Swap test:** the same procedure with the twin memory and without secondary golds, at middle depth.
8. **Statistics:** `build/stats.py`.

## Appendix B: Decision rules

- **Question inclusion:** the question must fail with no memory (style rule gold in at most 20% of 5 samples; task target tool used first in at most 30%; judge-graded at most 1 of 3; allergy questions need an allergen hit in at least 60%). It must also follow its gold and its twin in 4 of 4 tries under perfect retrieval.
- **Store exclusion:** remove any distractor Sonnet confirms supports the twin or contradicts the gold. Distractors that support the gold become secondary golds; drop questions with more than 3.
- **Tuning:** cutoffs are chosen on development questions only, as the cheapest setting that keeps the gold memory about as often as the most permissive one. Results: Jev groups 0.1, Jev memories 0.7, LLM picker 0.6, at most 20 memories.
- **Grading gate:** the Haiku judge must agree with Sonnet on at least 90% of a sample (it got 93%).
- **Cost cap:** a run refuses to start above 1.5× its estimate.
- **Verdicts:** a setup *matches* another when the paired 95% interval includes zero, and *beats* it when the interval is entirely above zero.

## Appendix C: Recipe, relevance-judged memory retrieval

A self-contained spec of the pattern this study found, for an agent or team to implement.

**Use it when** an agent has more memories than fit comfortably in a prompt, and some matter for requests they don't resemble: standing rules, constraints such as allergies, and preferences that apply to one kind of task.

**Components**

1. A memory store with each memory filed into a topic group: 20–30 groups for about 1,000 memories, each with a name and a one-sentence description.
2. A relevance judge: a fast classifier that answers yes/no questions with probabilities (Jev here), or an LLM that scores memories from 0 to 1. The classifier was as accurate and about 10× faster.
3. The answering model, with your normal prompt.

**For each message**

1. Ask the judge, for every group: "Could any memory in this group change or improve the reply to this message? Groups that apply to every reply always count." Keep groups with p ≥ 0.1.
2. Ask the judge, for every memory in the kept groups: "Could this memory change or improve the reply to this message?" Send up to 250 memories per call, and retry with backoff on rate limits.
3. Keep memories with p ≥ 0.7, highest first, at most 20.
4. Put them in the prompt under a fixed instruction ("Below are memories you have stored about this user ... Use them when they are relevant.") and answer.

**Defaults and tuning.** Start with 0.1, 0.7 and 20. Re-tune on held-out questions of your own with known gold memories, choosing the cheapest cutoffs that keep the gold memory about as often as the most permissive ones.

**Expected results at 1,000 memories:** gold memory in the prompt about 96% of the time; adherence of 93–94% on Haiku and Sonnet, matching or beating injecting all memories; about 9 memories (about 1.3k tokens) in the prompt; picking cost about $0.001 per message with Jev.

**How to evaluate it.** Use gold memories with a pass check per question, a no-memory baseline, and a swap test: replace each gold with a minimal twin and check that replies follow the twin.

**Known limits.** It has only been tested on one synthetic store of 1,000 memories, with conflicts screened out. The judge's work grows with group size: for large stores, split groups past a size limit and walk the tree, or cap candidates per group with a cheap first-pass search. Near-miss memories (related but irrelevant) are untested.

## Reproduce

Credentials come from the SDK's normal lookup (for example an `ant auth login` profile). The OpenAI embedding key and the Jev key are added to outgoing requests by the environment's proxy; the code sends a placeholder header. No keys are read from or stored in this repository.

Build and freeze the test material:

```bash
uv sync
uv run python -m build.style_rules      # self-test of the regex checks
uv run python -m build.generate         # Opus generation (resumable)
uv run python -m build.screen           # no-memory screens
uv run python -m build.finalize         # merge into data/items_v1.jsonl
uv run python -m build.validate_oracle  # perfect-retrieval validation
uv run python -m build.conflicts        # conflict screen
uv run python -m build.screen_all       # every gold against all 1,000 memories
uv run python -m build.store            # freeze store, items and hashes
uv run python -m build.ontology         # groups, entries and filing
```

Tune the cutoffs on the 75 development questions:

```bash
uv run python -m build.jev_tune         # Jev group cutoff
uv run python -m build.jev_memories     # Jev memory cutoff
uv run python -m build.llm_pick         # LLM picker cutoff
```

Run and analyze (each plan takes `--estimate` first; runs are resumable and refuse to start above `--max-cost`):

```bash
uv run python -m build.trials main_haiku --batch
uv run python -m build.trials swap_haiku --batch
uv run python -m build.trials main_sonnet --batch
uv run python -m build.trials jevplus_haiku --batch
uv run python -m build.trials jevplus_sonnet --batch
uv run python -m build.trials jevplus_swap_haiku --batch
uv run python -m build.trials llmpick_haiku --batch
uv run python -m build.trials llmpick_sonnet --batch
uv run python -m build.stats haiku
uv run python -m build.stats sonnet
uv run python -m build.report main_haiku
```

The pilot (`uv run python -m build.run pilot`) was a small Haiku run of every setup before the main runs.

## Files

| Path | Contents |
|---|---|
| `build/config.py` | Models, prices, store composition, allergens, mock tools |
| `build/style_rules.py` | The 20 hand-written style rules, twins and regex checks |
| `build/generate.py` | Opus generators for every other kind |
| `build/harness.py` | Prompt template, model calls (with tool rounds), judge |
| `build/grade.py` | One grader for every question type |
| `build/screen.py`, `build/validate_oracle.py`, `build/conflicts.py`, `build/screen_all.py` | The filters, including the exhaustive gold screen |
| `build/finalize.py`, `build/store.py` | Merge and freeze |
| `build/search.py` | The shared hybrid search: BM25 + OpenAI `text-embedding-3-small`, reciprocal rank fusion, top 10, tie-safe ranks |
| `build/ontology.py`, `build/types.py` | The 26-group ontology and the 4 type entries; Haiku files every memory |
| `build/jev_tune.py`, `build/jev_memories.py`, `build/llm_pick.py` | Jev group scores, Jev memory scores, the Haiku picker, and their tuning |
| `build/run.py` | The pilot runner and the setups' selection policies |
| `build/trials.py` | Main runs: plans, prompt caching, Message Batches, swap test |
| `build/stats.py`, `build/report.py` | Bootstrap intervals and paired differences; per-run summaries and cost |
| `data/items_final.jsonl`, `data/store_v1.json`, `data/frozen.meta.json` | The frozen questions, store and hashes |
| `data/ontology_v1.json`, `data/ontology_types.json`, `data/embeddings/` | The ontology, the type entries and cached embeddings |
| `data/gen/`, `data/screen/`, `data/validate/`, `data/screen_all/` | Every generated item, screen and validation reply, with grades |
| `data/jev_tune/`, `data/jev_memories/`, `data/llm_picks/` | Development split, cached picker scores and tuning summaries |
| `data/runs/*.jsonl` | Every trial: settings, memories shown, token usage, grades and the full reply |
| `data/stats_haiku.json`, `data/stats_sonnet.json` | The numbers in the Results table |
