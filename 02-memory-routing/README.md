# 02 · Memory routing (in progress)

**Question:** should an agent's memories be organized (searched by group, pinned by type, or chosen by reasoning over an ontology) rather than searched all together or injected all at once? The measure is how often the right memory gets used, and at what token cost.

Status: stage 1 (build and freeze the test material) is done. The setups, pilot and main run come next. Design and hypotheses live in the study brief.

## Stage 1 results

| Step | Result |
|---|---|
| Generation (`claude-opus-5-5`) | Gold memories with swap twins and pass rules, allergy queries, task procedures over a 20-tool mock catalog, decoys, past-session distractors |
| No-memory screen (`claude-haiku-5-5`) | Questions Haiku passes without memory are dropped. Where Haiku's default already matched the generated gold but not the twin, the two were swapped (`"swapped": true`) |
| Perfect-retrieval validation | Each question runs with only its gold memory (x2) and only its twin (x2); kept only if all four replies follow the memory given |
| Judge check | Haiku judge agreed with Sonnet on 56 of 60 sampled replies; allergy replies that hit the word list are judged by Sonnet |
| Conflict screen | 243 distractors that contradicted or duplicated a gold memory were removed |
| Frozen | 1,000-memory store in 5 fixed shuffles and 265 questions, hashed in `data/frozen.meta.json` |

Final questions: allergy 51, task procedures 45, unclear type 40, no memory 40, past sessions 38, facts 34, style rules 17. Store: 600 semantic, 250 episodic, 150 procedural (36 task decoys, 40 generated every-reply rules, 74 content-choice rules from study 01's `haystack_v1`). The every-reply rules were added after the first ontology build showed the store had no rules that apply to every reply, which would have made the ontology unable to say so.

Stage 1 cost about $6 in API calls, including re-runs after the grading fixes below.

## Grading decisions made during stage 1

- **Style rules, first match wins:** "21.1 km (13.1 miles)" follows the metric rule; "13.1 miles (21.1 km)" follows the imperial twin.
- **Task procedures, tool order:** the model gets up to 3 tool rounds with stub results and is told to act on reasonable assumptions. It passes if, of the gold and twin tools, the gold one is used first.
- **Judge on intent:** the judge ignores preambles and placeholders and doesn't count foods listed under "avoid" as recommended.

## Reproduce

Credentials come from the SDK's normal lookup (for example an `ant auth login` profile). No keys are read from or stored in this repository.

```bash
uv sync
uv run python -m build.style_rules      # self-test of the regex checks
uv run python -m build.generate         # Opus generation (resumable)
uv run python -m build.screen           # no-memory screens
uv run python -m build.finalize         # merge into data/items_v1.jsonl
uv run python -m build.validate_oracle  # perfect-retrieval validation
uv run python -m build.conflicts        # conflict screen
uv run python -m build.store            # freeze store, items and hashes
uv run python -m build.ontology         # groups, entries and filing
```

## Files

| Path | Contents |
|---|---|
| `build/config.py` | Models, prices, store composition, allergens, mock tools |
| `build/style_rules.py` | The 20 hand-written style rules, twins and regex checks |
| `build/generate.py` | Opus generators for every other kind |
| `build/harness.py` | Prompt template, Haiku calls (with tool rounds), judge |
| `build/grade.py` | One grader for every question type |
| `build/screen.py`, `build/validate_oracle.py`, `build/conflicts.py` | The three filters |
| `build/finalize.py`, `build/store.py` | Merge and freeze |
| `build/search.py` | The shared hybrid search: BM25 + OpenAI `text-embedding-3-small`, reciprocal rank fusion, top 10, tie-safe ranks |
| `build/ontology.py` | Opus proposes the groups and entries; Haiku files every memory, gold included |
| `data/ontology_v1.json`, `data/embeddings/` | The ontology and cached embeddings |
| `data/gen/` | Every generated item |
| `data/screen/`, `data/validate/` | Every screen and validation reply, with grades |
| `data/items_final.jsonl`, `data/store_v1.json` | The frozen questions and store |
