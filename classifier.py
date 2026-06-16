import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    instructions = (
        "You are classifying podcast episodes by their structural format "
        "(not their topic). Classify the episode into EXACTLY ONE of these four labels:\n\n"
        "- interview: a host draws out one or more guests; a clear host-asks / guest-answers dynamic.\n"
        "- solo: a single host speaking alone from memory, experience, or opinion — no guests, "
        "no assembled external sources.\n"
        "- panel: three or more speakers of roughly equal standing discussing or debating a shared "
        "topic; nobody is the single subject.\n"
        "- narrative: a story assembled from external sources (reporting, documents, archives, "
        "interview clips) with a clear story arc.\n\n"
        "Below are labeled examples. Learn the pattern, then classify the final episode.\n"
    )

    example_blocks = []
    for ex in labeled_examples:
        example_blocks.append(
            f"Title: {ex.get('title', '').strip()}\n"
            f"Description: {ex.get('description', '').strip()}\n"
            f"Label: {ex['label']}"
        )
    examples_section = "\n\n---\n\n".join(example_blocks)
    if examples_section:
        examples_section = (
            "### Labeled examples\n\n" + examples_section + "\n\n---\n\n"
        )

    target_section = (
        "### Episode to classify\n\n"
        f"Description: {description.strip()}\n\n"
        "Classify the episode above. Respond in EXACTLY this format and nothing else:\n\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one or two sentences explaining the choice>"
    )

    return f"{instructions}\n{examples_section}{target_section}"


def _parse_response(text: str) -> dict:
    """Extract a (lowercased) label and reasoning from the LLM's text response."""
    label = None
    reasoning = None
    leftover = []

    for line in text.splitlines():
        # Strip surrounding whitespace and leading markdown noise (e.g. "**", "- ")
        # so "**Label:**", "- Label:", "Label:" all match.
        stripped = line.strip()
        bare = stripped.lstrip("*#->` ").strip()
        low = bare.lower()
        if label is None and low.startswith("label:"):
            label = bare.split(":", 1)[1]
            # Clean surrounding whitespace, markdown, quotes, and punctuation.
            label = label.strip(" \t*`\"'.,").lower()
        elif reasoning is None and low.startswith("reasoning:"):
            reasoning = bare.split(":", 1)[1].strip().strip("*`").strip()
        else:
            leftover.append(stripped)

    if not reasoning:
        # Fall back to whatever else the model said.
        reasoning = " ".join(l for l in leftover if l).strip() or "No reasoning provided."

    return {"label": label, "reasoning": reasoning}


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful podcast-format classifier. You answer with "
                        "exactly one of the four valid labels and a brief reason."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:  # network error, rate limit, etc.
        return {"label": "unknown", "reasoning": f"Classification failed: {exc}"}

    parsed = _parse_response(text)
    label = parsed["label"]

    # Validate: anything not in VALID_LABELS becomes "unknown".
    if label not in VALID_LABELS:
        return {
            "label": "unknown",
            "reasoning": parsed["reasoning"] or f"Unrecognized label in response: {text[:120]!r}",
        }

    return {"label": label, "reasoning": parsed["reasoning"]}
