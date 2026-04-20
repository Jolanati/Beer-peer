# Wine Peer — What We Did

---

## Step 1 — Food image classifier (CNN)

We took **101,000 food photos** from the Food-101 dataset and trained a convolutional neural network to recognize what dish is in a photo — 101 possible categories, from pizza to sushi to steak.

We built two versions:

- A **custom CNN from scratch** — at least 3 convolutional blocks, trained entirely on Food-101
- A **ResNet-50 fine-tuned** — frozen backbone with a new classification head, transfer learning from ImageNet

Both versions were evaluated on the held-out test split and compared. Grad-CAM was used to visualize which part of the food photo drove each prediction.

---

## Step 2 — Wine review classifier (BiLSTM)

We took **824,000 real Vivino tasting notes** from the WineSensed dataset (NeurIPS 2023). Each review was written by a real Vivino user about a specific wine. We used the primary grape variety as the classification label.

We selected the **top 15 grape varieties by review count** — Cabernet Sauvignon, Merlot, Pinot Noir, Chardonnay, Sauvignon Blanc, and 10 others — covering ~85% of all reviews.

We trained two text classifiers:

- A **unidirectional LSTM** baseline
- A **bidirectional LSTM with attention** — the main model

The BiLSTM serves two roles:

1. **Classifier** — predicts grape variety from a tasting note (15 classes)
2. **Encoder** — at inference, finds the single most representative real Vivino review per grape to show the user as a tasting note

---

## Step 3 — Food flavor table (LLM-generated)

We used an LLM to generate a **food flavor table** — one entry per Food-101 dish, three rows each:

| Key | What it describes |
| --- | --- |
| `complement` | Tastes that echo and amplify the dish — e.g. pizza: *tomato, earthy, savory, rich* |
| `contrast` | Tastes that cut through and refresh — e.g. pizza: *sour, sharp, acidic, bright* |
| `balance` | Tastes that are crowd-safe and neutral — e.g. pizza: *light, creamy, soft, smooth* |

Keywords are written in plain taste and texture language — *tomato*, *smoky*, *creamy*, *sour* — not in wine vocabulary. This is intentional: the Word2Vec model handles the translation (see Step 4).

The flavor table is external curated knowledge. The ML models do not learn it — it is the bridge between what the CNN sees and what Word2Vec searches for.

---

## Step 4 — Word2Vec flavor embedding (pre-trained + fine-tuned)

We took a **pre-trained Word2Vec model** (Google News, 300-dimensional) — it already understands everyday food language: *tomato*, *fatty*, *smoky*, *sour*, *creamy* all have well-placed vectors.

We then **fine-tuned it on the 824,000 WineSensed reviews** using gensim — so wine-specific words (*Sangiovese*, *tannic*, *cassis*, *terroir*, *mineral*) were pulled into the same vector space as the food words.

For each of the 15 grape varieties, we averaged all word vectors from its reviews into a single **grape embedding vector**. The result: 15 points on the same flavor map as the food keywords.

At inference:

1. The food flavor table gives three keyword lists (complement / contrast / balance)
2. Each list is embedded by averaging its word vectors
3. Cosine similarity finds the closest grape embedding to each list
4. Three grape variety recommendations are returned — one per pairing intent

---

## Step 5 — Recommendation card assembly

For each recommended grape:

- The **WineSensed dataset** is filtered to that grape and sorted by rating → the highest-rated real wine bottle + vintage year is selected
- The **BiLSTM encoder** finds the most representative real Vivino review for that grape
- The rating is converted to a user approval percentage: `rating / 5.0 × 100`

The final output card shows:

- Food label + CNN confidence
- Three grape recommendations (Complement / Contrast / Balance)
- For each: real wine bottle name, real Vivino tasting note, real Vivino approval %

Nothing is generated — every word in the output comes from a real Vivino user.

---

## Step 6 — Joint model: food-wine compatibility (+10 bonus points)

We built a joint compatibility model on top of the two frozen encoders:

- **Positive pairs:** food image + wine review of a grape that Word2Vec matched to that food → label `1`
- **Negative pairs:** food image + wine review of a clearly incompatible grape → label `0`
- **Architecture:** frozen CNN encoder + frozen BiLSTM encoder → concatenate → small FC head → sigmoid

Only the FC head was trained. The joint model generalises beyond the hand-written flavor table — it can score food-grape pairs that were never explicitly curated.

The interesting experiment: run the joint model across all 101 × 15 combinations and find the top-5 unexpected high-scoring pairs — what the model learned about food-wine compatibility that we never wrote down.

---

## Summary

| What | How |
| --- | --- |
| Food recognition | CNN (scratch + ResNet-50) trained on Food-101 |
| Wine review understanding | BiLSTM trained on 824k real Vivino reviews, 15 grape classes |
| Flavor bridge | LLM-generated food flavor table — plain taste language |
| Grape matching | Word2Vec (Google News pre-trained, fine-tuned on WineSensed) |
| Output | Real wine bottle + real Vivino quote + real approval % |
| Bonus | Joint model learns food-wine compatibility from (image, review) pairs |
