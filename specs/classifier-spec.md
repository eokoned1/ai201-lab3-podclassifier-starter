# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
A two-line, key-prefixed plain-text format:

    Label: <one of: interview, solo, panel, narrative>
    Reasoning: <one or two sentences>

Why this format:
- Reliable to parse with simple string ops: split into lines, find the line
  starting with "Label:" / "Reasoning:", take the text after the colon.
- Resilient: even if the model adds a preamble or markdown, scanning for the
  "label:" prefix still finds the answer. A bare single-word answer would be
  fragile if the model adds any commentary.
- I considered JSON, but small instruct models occasionally emit invalid JSON
  (trailing commas, code fences), and json.loads() then throws — more failure
  modes than a forgiving prefix scan. The key:value form gives most of the
  structure with far less brittleness.
```

---

**Edge cases to handle in the prompt:**

```
- Empty labeled_examples: build_few_shot_prompt still produces a valid,
  zero-shot prompt — the taxonomy in the task instruction is enough for the
  model to classify. (In practice app.py/evaluate guard against this and warn
  the user to finish Milestone 1, but the prompt itself does not crash.)
- Very short / vague description: the prompt still asks for exactly one label;
  the model picks the closest format. The reasoning will note the uncertainty.
- The instruction pins the output to EXACTLY one of the four lowercase labels
  and tells the model to copy the label string verbatim, reducing the chance
  of synonyms ("monologue", "roundtable") that would fail validation.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Split the response text into lines. Scan for the first line whose lowercased,
stripped form starts with "label:" and take everything after the colon as the
raw label (strip whitespace, punctuation, and surrounding markdown/quotes,
then lowercase). Do the same for "reasoning:" to get the reasoning text.

If no "reasoning:" line is found, fall back to using the whole response (minus
the label line) as the reasoning so the user still sees the model's thinking.
```

---

**Step 4 — Validate the label:**

```
Check the parsed label against VALID_LABELS (exact, lowercase match). If it is
one of the four, keep it. If it is anything else — a synonym, an empty string,
or a label we never asked for — set label to "unknown". app.py already knows
how to render "unknown", and "unknown" never matches a ground-truth label so
it is correctly counted as wrong during evaluation.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call + parsing in try/except. Failures that can happen:
  - Network / API error, rate limit, or timeout from Groq.
  - Empty or None content in the response.
  - A response with no recognizable "Label:" line.

On any exception, return {"label": "unknown", "reasoning": "<short error note>"}
instead of raising. This keeps the 20-call evaluation loop alive — one bad
response degrades to a single wrong prediction rather than crashing the run.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: Marine Biologist Dr. Amara Diallo on What Coral Bleaching Actually Looks Like (e001)
Raw response text:
Label: interview
Reasoning: The episode features a host conversing with a single guest, Dr. Amara
Diallo, in a clear host-asks/guest-answers dynamic, which is characteristic of an
interview format. The description highlights the guest's expertise and the topics
they discuss, further supporting the classification as an interview.

The model stuck to my two-line "Label: / Reasoning:" format exactly, with no
preamble or markdown fences around it.
```

**How did you parse the label out of the response?**

```
I split the response into lines and looked for the first line that starts with
"label:" (after lowercasing and stripping leading markdown like ** or - so a
bolded "**Label:**" still matches). I take everything after the colon, strip the
surrounding whitespace/quotes/punctuation, and lowercase it. I do the same scan
for a "reasoning:" line; if there isn't one I fall back to using the rest of the
response as the reasoning so the user still sees something. After parsing I check
the label against VALID_LABELS and force anything else to "unknown".
```

**Did any episodes return `"unknown"`? If so, why?**

```
No. All 20 test episodes parsed to one of the four valid labels (I got 20/20),
so nothing fell through to "unknown". I still kept the validation + try/except in
place so a flaky API call or an off-format response degrades to a single wrong
prediction instead of crashing the whole evaluation loop.
```

**One thing about the output format that surprised you:**

```
I expected to fight with the formatting more than I did. With temperature=0 and a
firm "respond in EXACTLY this format" instruction, llama-3.3-70b returned the same
clean two-line shape every single time, which made the parsing far simpler than I
planned for. It convinced me that a forgiving key:value format plus a strict
prompt is more reliable here than asking for JSON.
```
