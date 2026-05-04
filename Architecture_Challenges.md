# Wine Peer — Architecture Challenges & Engineering Decisions

*A record of the real problems we hit, what we tried, why it failed, and what we decided.*
*Written to show the thought process behind the final pipeline, not just the final result.*

---

## Overview

Building the `recommend()` pipeline looked straightforward on paper: CNN identifies food → Word2Vec finds matching grape → BiLSTM retrieves a review. In practice, connecting three independently trained models exposed a chain of problems, each hiding behind the fix for the previous one.

---

## Challenge 1 — Confidence score: always 100%

### What we saw
Every recommendation card showed 100% confidence. Pizza: 100%. Steak: 100%. Sushi: 100%. The number was meaningless.

### Why it happened
We were sampling 2,000 reviews randomly from the full test set (all 15 grapes mixed together) and running them through the BiLSTM classifier. The grape with the most reviews in the sample wins with near-unanimous softmax votes — **winner-takes-all inflation**. The score is mathematically correct but useless as a quality signal.

### What we tried
Switched to the **median** confidence score (50th percentile of the sample distribution) instead of argmax. Honest, not inflated.

### Why that failed too
The median of a mixed-grape sample was around **0.7%** for some grapes. The target grape is a minority in a sample drawn from 15 classes. Even correct predictions are drowned out.

### Root cause
The BiLSTM softmax is calibrated to separate grapes from each other, not to produce meaningful confidence scores for recommendation quality. Using it as a confidence metric was the wrong tool for the job.

### What we did
Abandoned BiLSTM softmax entirely for the recommendation card. Instead:
- Run BiLSTM on **target-grape-only** test reviews
- Collect the 256-dimensional hidden state from each review
- Compute the centroid of those hidden states
- Return the review whose hidden state is closest to the centroid

This picks the review that best represents the *average BiLSTM encoding* of that grape's tasting language — genuinely central, no classifier scoring artefact.

The score on the card is now the **Word2Vec flavor match %** — a separate, honest signal from a different part of the pipeline.

---

## Challenge 2 — Three pairings, same grape every time

### What we saw
After fixing confidence, we noticed Safe Bet, Bold Move, and Hidden Gem all recommended the same grape variety. The 20-food table had almost no variety across rows.

### First diagnosis: KeyError on grape_class column
The column in `df_test` was `grape_class`, not `grape_variety`. Minor fix.

### What we tried first
A **keyword exclusion workaround**: after finding the first grape, pass it to `exclude=` and search again for the second, then a third. Three distinct grape names returned on every row.

### Why that was wrong
The three grapes were still semantically clustered — forced to be different labels but all from the same region of the embedding space. Pizza's "Safe Bet" and "Bold Move" were Sangiovese vs. Barbera — both Italian reds, both in the same corner of the space. The exclusion workaround masked the symptom without touching the cause.

### Root cause: two compounding problems discovered

**Problem A — Food vocabulary is not wine vocabulary**

The food flavor table is written in natural food language: `cheesy`, `baked`, `fatty`, `starchy`, `greasy`. These words exist in the Google News base Word2Vec but **drift to meaningless regions** after fine-tuning on 824k wine reviews — wine reviewers never write those words.

The `_WINE_VOCAB` filter (words appearing in both W2V and the training corpus) was silently dropping most food keywords, leaving only 1–2 words to represent each food dish. With 1–2 words, all foods land in roughly the same area of the embedding space.

**Problem B — Grape centroids are too tightly clustered**

Each grape centroid is the mean of thousands of word vectors from that grape's reviews. Averaging across many reviews collapses all 15 centroids toward the same "generic wine language" center. The 15 grapes form a tight ball rather than a spread-out sphere.

Concrete evidence: the **Bold Move contrast % was only 7–18%**. If the most opposite grape is only 7% different from the Safe Bet, all 15 centroids are essentially at the same point.

---

## Challenge 3 — Fixing the food side: query expansion

### Design principle established
The food flavor table is **correctly** written in food language. That is intentional and should stay. Curators write `cheesy`, `baked`, `fatty` because that is how they naturally think about food. The system's job is to translate — not to require the JSON to speak in wine vocabulary.

### Solution: query expansion as a translation layer

For each food keyword:
1. Check if it exists in wine vocabulary (keep it if yes)
2. Find its nearest Word2Vec neighbours **within the wine vocabulary**
3. The W2V model itself does the translation:

| Food keyword | In wine vocab? | Expands to |
|---|---|---|
| `cheesy` | ✗ | *buttery, creamy, rich, lactic, oaky* |
| `baked` | ✗ | *toasty, roasted, warm, caramelised, spiced* |
| `tomato` | ✓ | *cherry, raspberry, redcurrant, plum, cassis* |
| `fatty` | ✗ | *oily, rich, full-bodied, heavy, buttery* |
| `herby` | ✓ | *herbaceous, garrigue, thyme, sage, rosemary* |

The food JSON stays in food language. Expansion is automatic, data-driven, and not hard-coded.

### Result after query expansion
Better variety in the table — different foods now land in different regions. But Barbera and Tinta Roriz still dominated most rows, and Bold Move contrast was still very low (7–18%).

**Conclusion: expansion fixed the food side. The grape side (Problem B) still needs fixing.**

---

## Challenge 4 — Fixing the grape side: mean-centering *(current step)*

### The geometry problem
All 15 grape centroids sit in a tight cluster. When you measure cosine similarity from a food query to each centroid, the differences are too small to be meaningful — every food query is roughly equidistant from all grapes.

### Solution: mean-centering

Subtract the global mean of all 15 grape centroids from each centroid before computing similarities:

```
centered_centroid[i] = grape_centroids[i] - mean(grape_centroids)
```

This repositions the origin at the center of the grape cluster. The 15 centroids now point outward in different directions from that center, making angular differences large and meaningful.

Think of it as spreading 15 people from a tight crowd into a wide circle — from far away you couldn't tell them apart, but once they spread out, you can clearly point to the one nearest to you.

**Expected outcome:** Bold Move contrast % should jump from 7–18% to 40–60%+. Different foods should land near genuinely different grapes.

### Why not Option B (discriminative centroids) first?
Option B (per-grape TF-IDF weighting) rebuilds centroids so each one emphasizes what is linguistically *unique* to that grape. More principled, but more work. Mean-centering is a one-line fix that costs nothing and is completely reversible. The right order is: fix the geometry first, then decide if the centroids themselves need to be rebuilt.

---

## Challenge 5 — Score display: building user trust *(pending)*

### The problem
Raw cosine similarity values are not intuitive to users:
- 13% for Hidden Gem looks like a failure
- 70% for Safe Bet looks good but is hard to compare to 41%

### Plan
After centroid geometry is fixed, rescale similarity scores to a display range that feels credible (e.g. 55–95%), using the actual min/max of the distribution across all 20 test foods rather than a hard-coded formula (which would produce the same number for everything — a bug we already hit and fixed).

---

## Challenge 6 — Product recognisability *(pending)*

### The problem
Even with correct variety in recommendations, some grapes are unfamiliar to most users:
- **Tinta Roriz** is legitimate (it is Tempranillo in Portugal) but unknown outside specialist circles
- Users expect to see Pinot Noir, Chardonnay, Sauvignon Blanc

### Options under consideration
- Display grape synonym names alongside the technical variety name
- No algorithmic change needed — a display/labeling decision

---

## Summary of decisions made

| Problem | Symptom | Root cause | Fix applied |
|---|---|---|---|
| Confidence = 100% | All cards show 100% | Argmax on mixed-grape sample | BiLSTM hidden-state centroid |
| Confidence = 0.7% | All cards show near-zero | Target grape minority in mixed sample | Abandoned BiLSTM softmax entirely |
| Same grape all three | No diversity | Missing `grape_class` column name | Column name fix |
| Same grape all three | No diversity | Keyword exclusion workaround | Removed — masked symptom only |
| Same grape all three | No diversity | Food keywords drop out of wine vocab | Query expansion translation layer |
| Barbera/Tinta Roriz always | No variety | Grape centroids too clustered | Mean-centering *(in progress)* |
| Low contrast % (7–18%) | Bold Move not bold | Same clustering problem | Mean-centering *(in progress)* |
| Unintuitive scores | 13% looks like failure | Raw cosine sim displayed directly | Score rescaling *(pending)* |
| Unknown grape names | Not product-ready | Technical variety names | Display synonym mapping *(pending)* |

---

*Architecture Challenges v1.0 — Wine Peer, RSU Advanced Machine Learning Course*
