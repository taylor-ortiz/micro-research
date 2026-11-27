# Memory routing post (draft)

## 1. Why I looked at this (locked)

A recent study I published showed that for smaller models, where a memory sits in a list of memories decides whether the model finds it and applies it. With about 1,450 stored preferences, Claude Haiku 5.5 followed one listed first 98–99% of the time, and one in the middle 0% of the time. Larger models had no trouble. That led to the next question: when an agent answers a message, how should it choose which memories go into the prompt? Does organizing the memories help it find the right ones? And what does each approach cost in tokens?

## Why not just inject everything? (locked)

The natural assumption is to put every memory straight into the prompt alongside the user's message, so the right one is always there to apply. There are four problems with that. It uses a ton of tokens: about 66k per question in this study. More tokens to read means more latency. And eventually your memories outgrow the context window, especially for something like account-level memory, which forces you to decide which ones to keep anyway.

The last problem, and arguably the most significant, is that full injection doesn't guarantee the model will use the right memory. On Haiku, injecting all 1,000 memories got it right 87% of the time, and memories in the middle of the list did worse than ones at the top.

## Three kinds of memory (locked)

Before the results, here are the kinds of memory I used. The test user had 1,000 of them.

- **Semantic:** facts about the user. "Allergic to peanuts," "vegetarian." (600)
- **Episodic:** past conversations that give context later. "On Sept 17 we drafted the grant proposal; the evaluation plan is still left." (250)
- **Procedural:** how the agent should act. Some are standing rules that apply to every reply, like "use metric units." Others are the steps for a task, like which tool to try first when booking a hotel. (150)

The types matter because they look different from the questions they apply to, and that turned out to decide everything.

## 2. What I found (locked)

Using a fast classifier model to pick the memories that matter for each message worked as well as injecting all 1,000. It picked about 9 memories per question and got it right 93% of the time on Haiku and 94% on Sonnet, against 87% and 96% for full injection, with about 98% fewer tokens for the answering model. Plain similarity search, the usual way agents fetch memories, also puts about 10 memories in the prompt, but got it right only 60–63% of the time.

![Accuracy vs. tokens the answering model reads](images/accuracy-vs-tokens.png)

## 3. What I expected (locked)

My original hypothesis was that it was better to organize memories than to search them all together. Standing rules like "use metric units" apply to every reply but share no words with any question, so I figured plain search would miss them. My theory was a memory ontology: sort every memory into topic groups, give each group a name, a description, when it applies and a few examples, and let the model read that like a table of contents to decide which groups to search. I assumed the more detail each group had, the better the model would choose.

Neither held up. The best organized setup came in about 15 points under full injection, and richer descriptions didn't help at all. Having a classifier judge each individual memory was a late addition after that first result set came in.

## 4. How I tested it

## 5. What happened

## 6. What I'd do with this

## 7. What I can't claim yet

---

## Notes for later sections (not part of the post)

- Section 6/7: full injection hits a hard wall at the context window. The classifier judges each memory independently, so the store can be split across calls; its cost grows with the store. Filtering by category first keeps that bounded. In this study the 0.1 group cutoff kept ~530 of 1,000 memories on average (range 253–949), so finer groups (a growing ontology that splits as it fills) are needed to really bound it.
- Section 7: isolation means the classifier can't spot conflicting or duplicate memories; the test store was screened for conflicts.
- Section 7: untested: classifier with no group step; richer group entries for the classifier; group context when judging a memory.
