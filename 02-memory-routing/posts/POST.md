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

## 2. What I found

## 3. What I expected

## 4. How I tested it

## 5. What happened

## 6. What I'd do with this

## 7. What I can't claim yet
