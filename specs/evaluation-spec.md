# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = (number of positions where prediction == ground_truth) / (total positions)

A prediction is "correct" when it EXACTLY matches the ground-truth label at the
same index. We divide by the total number of predictions (= number of episodes).
```

---

**Step-by-step logic:**

```
1. If there are no predictions (empty list), return 0.0 (avoid divide-by-zero).
2. Count the positions where predictions[i] == ground_truth[i].
3. Divide that count by len(predictions).
4. Return the result as a float between 0.0 and 1.0.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. There is nothing to score, and dividing by zero is undefined — 0.0
is a safe, meaningful "no correct predictions" value that won't crash the report.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

index 0: interview == interview  ✓
index 1: solo      == solo       ✓
index 2: panel     != solo       ✗
index 3: interview != narrative  ✗

correct = 2, total = 4  ->  2 / 4 = 0.5
compute_accuracy() returns 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For class C, an episode counts as correct when its ground-truth label is C AND
the prediction also equals C. (This is recall for class C.) Example: for
"interview", correct = episodes whose true label is "interview" that were
predicted "interview". We only look at episodes whose TRUTH is C — predicting
"interview" for a truly-solo episode does NOT add to interview's "correct".
```

---

**What does "total" mean for a given class?**

```
total = the number of episodes whose GROUND-TRUTH label is C — i.e. how many of
this class actually exist in the test set. It is NOT the total number of
predictions, and NOT the number of times we predicted C. This makes accuracy =
correct/total a per-class recall.
```

---

**Step-by-step logic:**

```
1. Initialize a stats dict: for each label in VALID_LABELS,
   {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over the (predicted, truth) pairs with zip(predictions, ground_truth).
3. For each pair: if truth is a valid label, increment stats[truth]["total"];
   and if predicted == truth, also increment stats[truth]["correct"].
4. After the loop, for each label compute accuracy = correct / total,
   or 0.0 if total == 0.
5. Return the stats dict keyed by label.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0 (per the docstring in evaluate.py). With no examples there
is no meaningful rate to report, and 0.0 avoids a divide-by-zero — it's a
sentinel meaning "not measured", not a real 0% score.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Walk through by truth label:
- interview: truth at idx 0; pred 0 = interview ✓  -> correct 1, total 1
- solo:      truth at idx 1,2; pred 1=interview ✗, pred 2=solo ✓ -> correct 1, total 2
- panel:     truth at idx 3; pred 3 = panel ✓  -> correct 1, total 1
- narrative: truth at idx 4; pred 4 = panel ✗  -> correct 0, total 1

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.00
solo           1       2      0.50
panel          1       1      1.00
narrative      0       1      0.00
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
