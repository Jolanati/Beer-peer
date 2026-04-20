# Wine Peer

Deep Learning Project - RSU

*Photograph your food. Discover your perfect wine.*

---

## 1. Project Idea

Wine Peer is a deep learning application that bridges the gap between what a person is eating and what they should be drinking. The core concept is simple: a user photographs their meal, and the app returns three wine recommendations — one that complements the food's flavor, one that contrasts it, and one that balances it — each with a real tasting note from a Vivino user and an approval percentage.

The motivation comes from a genuine everyday moment. You sit down at a restaurant, or you are cooking at home, and you wonder what to drink. Wine pairing guides exist but are rarely accessible at the moment of decision. Wine Peer makes that knowledge available through a single photograph.

The project is academically grounded in two independent machine learning tasks — image classification and text classification — connected by a flavor embedding layer. A CNN recognises the food, Word2Vec maps food flavor vocabulary to grape variety vocabulary, and a BiLSTM retrieves genuine tasting language from 824,000 real Vivino reviews. This architecture satisfies the dual-dataset and CNN+LSTM requirement of the project rubric while producing a product concept with genuine real-world utility.

---

## 2. Product

### 2.1 User Flow

The interaction is designed to take under 10 seconds from input to output:

1. User opens Wine Peer and photographs their meal on a plate or table.
2. The CNN model identifies the dish from the image.
3. The food's flavor profile (a set of taste-descriptor keywords in wine vocabulary) is looked up from the food flavor table.
4. Word2Vec — trained on 824,000 real Vivino reviews — maps those keywords into the same flavor embedding space as grape varieties. Cosine similarity finds the best match for each of three pairing intents: Complement, Contrast, Balance.
5. The BiLSTM encoder retrieves the single most representative real Vivino review for the matched grape variety.
6. The user sees the dish name, three wine recommendations (one per intent) with grape variety, a real wine bottle name, a genuine drinker quote, and a Vivino approval percentage.

### 2.2 Example Output

| | |
| --- | --- |
| **Input** | Photo of pizza margherita |
| **CNN output** | Pizza (94% confidence) |
| **Food flavor profile** | savory · salty · cheesy · rich · fatty · tomato |

| Pairing | Grape | Wine | Drinker Quote | Vivino |
| --- | --- | --- | --- | --- |
| **Complement** — amplifies the food's flavors | Sangiovese | Chianti Classico Riserva 2019 | *"Deep cherry and dried herbs — wraps around the tomato sauce like it was made for it."* | 92% users |
| **Contrast** — cuts through and refreshes | Chardonnay | Chablis Premier Cru 2021 | *"Bone-dry mineral acidity cuts straight through the richness. Resets every bite."* | 89% users |
| **Balance** — neutral crowd pleaser | Pinot Grigio | Santa Margherita Alto Adige 2022 | *"Light, clean and gently fruited. Gets out of the way and lets the pizza do the talking."* | 86% users |

---

## 3. Datasets

### 3.1 Image Dataset — Food-101

| | |
| --- | --- |
| **Name** | Food-101 |
| **Source** | ETH Zurich / torchvision |
| **Load command** | `torchvision.datasets.Food101(root=DATA_DIR, download=True)` |
| **Size** | 101,000 images across 101 food categories |
| **Classes** | 101 dishes: pizza, sushi, steak, burger, ramen, curry, salad, pasta and more |
| **Split** | 750 training images and 250 test images per class |
| **Format** | RGB images, variable resolution, pre-labeled by class folder |

Food-101 is one of the most widely used food image classification benchmarks in computer vision research. Its class labels map directly and naturally to flavor profiles, enabling a clean handoff to the pairing logic layer.

### 3.2 Text Dataset — WineSensed (Vivino Reviews)

| | |
| --- | --- |
| **Name** | WineSensed — Learning to Taste (NeurIPS 2023) |
| **Source** | Hugging Face Hub |
| **Load command** | `load_dataset("Dakhoo/L2T-NeurIPS-2023", "vintages", trust_remote_code=True)` |
| **Size** | 824,000 real Vivino user tasting notes tied to 350,000+ unique wine vintages |
| **Key columns** | `review` (tasting note), `wine` (name), `year` (vintage), `grape` (variety), `rating` (Vivino avg 0–5), `country`, `region` |
| **License** | CC BY-NC-ND 4.0 — non-commercial research use |

WineSensed is a NeurIPS 2023 multimodal wine dataset built from Vivino data. Every review is a genuine human-written tasting note. The primary grape variety (first entry in the `grape` column) serves as the BiLSTM classification label. The top 15 grape varieties by review count are used as the 15 classification classes, covering approximately 85% of all reviews.

---

## 4. High-Level Architecture

The project pipeline consists of four sequential layers. Each layer is independently trained and evaluated before being connected into the full pipeline.

```text
PHOTO → CNN → food label → food flavor profile → Word2Vec similarity → grape variety (Complement / Contrast / Balance) → BiLSTM review retrieval → flavor language + rating %
```

| Step | Component | Function | Dataset |
| --- | --- | --- | --- |
| 1 | CNN — Image Classifier | Takes a food photograph as input. Outputs a food category label from 101 classes. | Food-101 |
| 2 | Food Flavor Table | Each of the 101 food classes has three sets of flavor keywords in wine vocabulary: `complement`, `contrast`, `balance`. | Embedded in notebook |
| 3 | Word2Vec — Flavor Embedding | Trained on WineSensed Vivino review text. Maps food flavor keywords into the same vector space as grape variety review language. Cosine similarity returns the closest grape per pairing intent. | WineSensed |
| 4 | BiLSTM — Review Retrieval | Trained on WineSensed reviews for 15-class grape classification. At inference, the encoder finds the most representative real Vivino review per recommended grape. The highest-rated wine of that grape is selected. | WineSensed |
| 5 | Output Layer | Combines CNN label + Word2Vec pairings + BiLSTM-retrieved Vivino quote + wine name + rating percentage into a structured recommendation card. | Combined |

### 4.1 CNN Architecture

The CNN uses a transfer learning approach based on ResNet-50, with the classification head fine-tuned on Food-101. Transfer learning is chosen because the base model carries strong general visual features from ImageNet training, significantly reducing the training time and data required to reach competitive accuracy on food classification.

- Input: RGB food image, resized to 224×224
- Backbone: ResNet-50 pre-trained on ImageNet (frozen in early epochs)
- Classification head: Linear layer with 101 output units, softmax activation
- Output: Food category label with confidence score

A CNN trained from scratch (3 convolutional blocks) is also implemented for comparison.

### 4.2 LSTM Architecture

The BiLSTM is trained on WineSensed review text to classify text into 15 grape variety categories. The model learns to associate descriptive language — *"cassis and cedar"* for Cabernet Sauvignon, *"strawberry and forest floor"* for Pinot Noir, *"mineral and citrus"* for Riesling — with grape labels. At inference time, the trained encoder retrieves the most representative real Vivino review per recommended grape: the review whose hidden-state vector sits closest to the grape centroid. No text is generated; all quotes shown to the user are genuine Vivino language.

- Input: Tokenised Vivino review text, padded to fixed length
- Embedding: GloVe-100d pre-trained word vectors
- Recurrent layer: Bidirectional LSTM (128 units, output 256-d)
- Classification output: 15 grape variety classes (training task)
- Inference mode: Grape-conditioned representative review retrieval

A unidirectional LSTM is also trained as a baseline for comparison.

### 4.3 Word2Vec Pairing

We start from Google's pre-trained Word2Vec (trained on ~100 billion words of Google News). This means the model already understands everyday food language — *tomato*, *spicy*, *fatty*, *smoky* all have well-placed vectors before any wine-specific training begins.

We then fine-tune this model on all 824k WineSensed Vivino reviews. Fine-tuning adds wine-specific vocabulary (*Sangiovese*, *tannic*, *cassis*, *terroir*) and repositions existing words so their neighborhoods reflect how wine reviewers use them. The result is a single vector space where food words and wine tasting words sit on the same map.

Grape embeddings are computed by averaging all word vectors across reviews for each of the 15 grape varieties. At inference, food flavor keywords (which can now be natural food words like *tomato* or *fatty*) are embedded the same way and matched to grape vectors by cosine similarity.

| Pairing intent | Keyword direction | What gets returned |
| --- | --- | --- |
| **Complement** | Keywords that echo the food's dominant flavor | Grape whose review language most shares the food's taste profile |
| **Contrast** | Keywords that cut through and refresh | Grape whose review language is most different but complementary (high acidity, mineral) |
| **Balance** | Light, approachable, crowd-safe keywords | Grape closest to clean / fruity / gentle in embedding space |

### 4.4 Food Flavor Table

The food flavor table is the bridge between the CNN's food label and the Word2Vec pairing system. It is a Python dictionary embedded directly in the notebook (no external file dependency). Each of the 101 Food-101 food classes has three sets of flavor keywords in grape-specific wine vocabulary:

```python
"pizza": {
    "complement": ["earthy", "savory", "cherry", "leather", "tannic", "herbaceous"],
    "contrast":   ["mineral", "crisp", "citrus", "steely", "dry", "acidic"],
    "balance":    ["light", "fruity", "clean", "gentle", "soft", "easy"],
}
```

Because we fine-tune from Google News Word2Vec, the flavor table can use natural food language — *tomato*, *fatty*, *smoky*, *creamy* — alongside wine tasting words. All of these have meaningful vectors in the fine-tuned model.

### 4.5 Why Grape Classification Rather Than Wine Types

Classifying by grape variety (Cabernet Sauvignon, Pinot Noir, Chardonnay…) rather than broad wine type (Red / White / Rosé) makes the text classification task significantly harder and more meaningful:

- The model must learn fine-grained flavor language — *"cassis and cedar"* vs *"strawberry and silk"* — not just colour-level signals.
- The output is more specific and useful: *"you'd like a Sangiovese"* rather than *"you'd like a Red wine"*.
- Word2Vec embeddings per grape place varieties in a nuanced flavor space where Syrah and Malbec cluster together but far from Riesling.

**The 15 grape classes** (selected by frequency, covering ~85% of all WineSensed reviews):

| Reds | Whites |
| --- | --- |
| Cabernet Sauvignon · Merlot · Pinot Noir · Syrah · Malbec · Sangiovese · Tempranillo · Grenache · Zinfandel | Chardonnay · Sauvignon Blanc · Riesling · Pinot Grigio · Viognier · Chenin Blanc |

---

## 5. Technical Stack

| | |
| --- | --- |
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch |
| **Image Processing** | torchvision, PIL |
| **NLP** | NLTK for preprocessing; gensim for Word2Vec; custom BiLSTM in PyTorch |
| **Data Loading** | Hugging Face `datasets` library |
| **Visualisation** | Matplotlib, Seaborn, WordCloud |
| **Deployment** | Streamlit on Hugging Face Spaces |

---

## 6. Success Criteria

- CNN achieves above 70% top-1 accuracy on Food-101 test set (state of the art with transfer learning reaches 90%+; 70% is the baseline target for this project scope).
- BiLSTM achieves above 65% accuracy in classifying Vivino review text by grape variety across 15 classes (15-class is a harder task than 8-class; 65% well above the 6.7% random baseline).
- The end-to-end pipeline produces a plausible grape recommendation for all 101 Food-101 classes covered by the flavor table.
- The system can process a single image input and return a full recommendation card in under 5 seconds on a standard laptop CPU.

---

Wine Peer - Project Description v1.0 - RSU Advanced Machine Learning Course
