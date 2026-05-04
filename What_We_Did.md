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
| `characteristic` | Tastes that echo and amplify the dish — e.g. pizza: *tomato, earthy, savory, rich* |
| `opposite` | Tastes that cut through and refresh — e.g. pizza: *sour, sharp, acidic, bright* |
| `unexpected` | A crowd-safe or surprising complement — e.g. pizza: *light, creamy, soft, honeyed* |

Keywords are written in plain taste and texture language — *tomato*, *smoky*, *creamy*, *sour* — not in wine vocabulary. This is intentional: the Word2Vec model handles the translation (see Step 4).

The flavor table is external curated knowledge stored in `data/food_flavor_table.json` (101 entries). The ML models do not learn it — it is the bridge between what the CNN sees and what Word2Vec searches for.

---

## Step 4 — Word2Vec flavor embedding (pre-trained + fine-tuned)

We took a **pre-trained Word2Vec model** (Google News, 300-dimensional) — it already understands everyday food language: *tomato*, *fatty*, *smoky*, *sour*, *creamy* all have well-placed vectors.

We then **fine-tune it on the 824,000 WineSensed reviews** using gensim — so wine-specific words (*Sangiovese*, *tannic*, *cassis*, *terroir*, *mineral*) are pulled into the same vector space as the food words. This is the key step that allows a food keyword like *tomato* to be compared directly against a grape variety’s tasting profile.

For each of the 15 grape varieties, we average all word vectors from its reviews into a single **grape centroid vector**. The result: 15 points (shape `15 × 300`) on the same flavor map as the food keywords, saved to `weights/grape_embeddings.npy`.

At inference:

1. The food flavor table gives three keyword lists (`characteristic` / `opposite` / `unexpected`)
2. Each list is embedded by averaging its word vectors
3. Cosine similarity finds the closest grape centroid to each list
4. Three grape variety recommendations are returned — one per pairing intent

---

## Step 5 — Recommendation card assembly

For each recommended grape:

- The **WineSensed dataset** is filtered to that grape and sorted by Vivino rating → the highest-rated real wine bottle + vintage year is selected
- The **BiLSTM encoder** scores a random sample of 2,000 test-set reviews; the review at the **median confidence score** (50th percentile) is shown as the most representative tasting note — avoiding argmax winner-takes-all inflation and uncalibrated softmax overconfidence
- The rating is converted to a user approval percentage: `rating / 5.0 × 100`

The final output card shows:

- Food label + CNN confidence
- Three grape recommendations (Characteristic / Opposite / Unexpected)
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

## Step 7 — DistilBERT bonus encoder (Sections 11.10–11.12)

As a bonus we added a **DistilBERT** text encoder (Sections 11.10–11.12). DistilBERT is a distilled Transformer — 40% smaller and 60% faster than BERT while retaining 97% of its language understanding.

We fine-tuned `distilbert-base-uncased` on the same 15-class grape classification task as the LSTM and BiLSTM. It serves as a comparison baseline to answer: does a pre-trained Transformer encoder outperform our BiLSTM on this wine review task, and by how much?

All three text models (LSTM, BiLSTM, DistilBERT) are compared side by side in Section 11.12. Results and weights are saved in Section 12 alongside the CNN models.

---

## Step 8 — Section 12: saving all results

Section 12 collects results from all 5 trained models and saves them to Google Drive:

- `weights/cnn_scratch.pt` — custom CNN
- `weights/cnn_resnet50.pt` — ResNet-50 fine-tuned
- `weights/lstm.pt` — LSTM baseline
- `weights/bilstm.pt` — BiLSTM with attention
- `weights/distilbert_best.pt` — DistilBERT fine-tuned

A `_sanitize()` function converts any tensors or numpy arrays in the results dicts to plain Python lists before saving, preventing `pickle.TypeError` on Colab.

A 5-model summary table is printed: model name, val accuracy, test accuracy, parameters.

---

## Engineering notes — Colab environment fixes

**Transformers import stability:** The standard Colab pre-install of `transformers` is pinned to an older version that conflicts with recent `huggingface-hub`. Importing directly causes a kernel crash. We fixed this by:

1. Using a subprocess probe (`subprocess.call([sys.executable, "-c", "from transformers import ..."])`) to detect whether the installed version is usable before attempting an import in the main kernel.
2. Pinning `transformers>=4.40.0,<5.0 tokenizers>=0.19,<1.0 safetensors>=0.4` and forcing a Colab runtime restart after install.
3. Removing `huggingface-hub` from the pinned block — pinning it to 0.36.2 was itself causing the crashes.

All pip installs are in Section 1.1; all library imports (including `DistilBertTokenizerFast`, `DistilBertModel`) are in Section 1.2 so the full environment is ready before any model cells run.

---

## Summary

| What | How |
| --- | --- |
| Food recognition | CNN (scratch + ResNet-50) trained on Food-101 |
| Wine review understanding | BiLSTM trained on 824k real Vivino reviews, 15 grape classes |
| Flavor bridge | LLM-generated food flavor table — plain taste keywords (characteristic / opposite / unexpected) |
| Grape matching | Word2Vec (Google News pre-trained, fine-tuned on WineSensed) |
| Grape centroids | 15 vectors (300-d each), one per grape class |
| Review retrieval | BiLSTM encoder — median-confidence review per grape (representative, not an argmax outlier) |
| Wine ranking | Highest Vivino rating per grape from df_wine |
| Output | Real wine bottle + real Vivino quote + real approval % |
| Bonus text model | DistilBERT fine-tuned on same 15-class task |
| Bonus joint model | Learns food-wine compatibility from (image, review) pairs |
