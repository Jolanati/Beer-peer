# Recommendations to Improve the Wine Peer Deep Learning Project

## 1. Keep the core idea

The core product idea is strong:

```text
User uploads food photo
        ↓
CNN recognises the food
        ↓
System recommends 3 wines
        ↓
Each wine includes bottle, review, rating, and explanation
```

This fits the assignment well because the project combines image analysis and text analysis in one business domain. The CNN handles the food image task, the BiLSTM/RNN handles the wine review text task, and the final recommendation layer combines both outputs into a practical user-facing decision.

---

## 2. Keep the CNN and BiLSTM branches

Do not restart the CNN or BiLSTM parts.

Keep this logic:

```text
CNN = recognises food from image
BiLSTM = works with wine review text and retrieves review evidence
```

The current problem is not mainly in the CNN or the BiLSTM.

The weak part is the middle layer:

```text
food label -> food profile -> wine recommendation
```

This is the part that needs to be redesigned.

---

## 3. Fix the naming in the food JSON

Currently, the JSON structure is confusing.

The metadata says:

```json
"classic": "The dish's dominant flavor profile"
"contrast": "The heavy, rich, fatty or intense qualities of the dish"
"safe_bet": "Neutral, crowd-pleasing qualities"
```

But the labels shown to the user are:

```json
"classic": "Safe Bet"
"contrast": "Bold Move"
"safe_bet": "Hidden Gem"
```

This means the key called `safe_bet` is actually displayed as `Hidden Gem`.

That creates confusion in the code and in the project explanation.

### Recommended change

Rename the JSON keys so they match the user-facing card names:

```json
"pizza": {
  "safe_bet": [...],
  "bold_move": [...],
  "hidden_gem": [...]
}
```

Use this meaning:

```text
safe_bet = classic, expected, reliable pairing
bold_move = contrast pairing that cuts through fat, spice, salt, sweetness, or richness
hidden_gem = less obvious but still compatible pairing
```

---

## 4. Stop using poetic phrases as ML keywords

The current JSON contains many expressive phrases, for example:

```text
electric acidity
wire taut
gossamer
ethereal
light as silk
warm opulent
taut nerve
piercing
searing
```

These sound nice for human writing, but they are not good model features.

Word2Vec splits these into separate words like:

```text
electric
warm
lean
taut
note
mineral
opulent
```

Many of these words repeat across many dishes, so the system starts seeing different foods as similar.

That is one reason why the same grapes appear again and again.

### Recommended change

Use simple, controlled descriptors.

Instead of this:

```json
"contrast": [
  "electric acidity",
  "wire taut",
  "piercing",
  "chalky",
  "searing"
]
```

Use this:

```json
"bold_move": [
  "high_acidity",
  "fresh",
  "mineral",
  "light_body",
  "cuts_fat"
]
```

For ML, boring and consistent is better than poetic.

---

## 5. Use a controlled vocabulary

The JSON currently mixes several types of words:

```text
ingredients: tomato, cheddar, bacon, dill pickle
cooking methods: baked, charbroiled, fried
wine terms: tannic, mineral, cedar, tobacco
emotional/style words: elegant, graceful, ethereal
places/cuisines: american classic, french bistro, japanese
```

This makes the embedding space noisy.

### Recommended change

Create a controlled vocabulary of food and wine descriptors.

Example food descriptors:

```text
fatty
creamy
salty
acidic
sweet
spicy
smoky
umami
fried
grilled
raw
delicate
rich
herbal
earthy
seafood
tomato_based
cheesy
dessert
```

Example wine descriptors:

```text
high_acidity
low_tannin
medium_tannin
high_tannin
light_body
medium_body
full_body
red_fruit
dark_fruit
citrus
mineral
floral
aromatic
oaky
buttery
peppery
earthy
fresh
```

Then each food should use these terms consistently.

---

## 6. Add structured food profile numbers

Currently, the food JSON only has keyword lists.

To make the recommender more useful, add numeric food features.

Example:

```json
"pizza": {
  "profile": {
    "intensity": 4,
    "fat": 3,
    "acid": 4,
    "salt": 3,
    "sweetness": 1,
    "spice": 1,
    "umami": 4
  },
  "safe_bet": [
    "tomato_based",
    "cheesy",
    "herbal",
    "red_fruit",
    "high_acidity"
  ],
  "bold_move": [
    "fresh",
    "mineral",
    "high_acidity",
    "cuts_fat"
  ],
  "hidden_gem": [
    "juicy_red_fruit",
    "soft_tannin",
    "easy_drinking"
  ]
}
```

This allows the system to reason more like a sommelier:

```text
If food is fatty -> wine needs acidity.
If food is spicy -> avoid high tannin.
If food is sweet -> avoid very dry wines.
If food is delicate -> avoid heavy, full-bodied wines.
If food is grilled or charred -> tannin, pepper, smoke, and dark fruit can work.
```

---

## 7. Add grape profiles

Do not rely only on Word2Vec to understand wine pairing.

Create a small profile dictionary for your grape classes.

Example:

```python
grape_profiles = {
    "Sangiovese": {
        "body": 3,
        "acid": 5,
        "tannin": 3,
        "sweetness": 1,
        "tags": [
            "red_fruit",
            "cherry",
            "herbal",
            "earthy",
            "tomato_based"
        ]
    },

    "Cabernet Sauvignon": {
        "body": 5,
        "acid": 3,
        "tannin": 5,
        "sweetness": 1,
        "tags": [
            "dark_fruit",
            "cedar",
            "tobacco",
            "grilled_meat",
            "high_intensity"
        ]
    },

    "Riesling": {
        "body": 2,
        "acid": 5,
        "tannin": 0,
        "sweetness": 2,
        "tags": [
            "citrus",
            "mineral",
            "floral",
            "aromatic",
            "spicy_food"
        ]
    },

    "Sauvignon Blanc": {
        "body": 2,
        "acid": 5,
        "tannin": 0,
        "sweetness": 1,
        "tags": [
            "citrus",
            "green_herb",
            "mineral",
            "fresh",
            "seafood"
        ]
    }
}
```

This does not make the project less machine-learning-based.

It creates a business logic layer that makes the final recommendation more stable, explainable, and useful.

---

## 8. Change the recommendation logic

The current logic is too close to this:

```text
food keywords -> Word2Vec similarity across all grapes -> choose top grapes
```

This is risky because some grapes become default winners.

### Recommended logic

Use candidate generation first, then ranking.

```text
CNN predicts food
        ↓
Food JSON gives structured taste profile
        ↓
Rule-based pairing engine creates candidate grapes
        ↓
Word2Vec adjusts semantic match score
        ↓
BiLSTM retrieves real review and confidence
        ↓
Final business score ranks Safe Bet, Bold Move, Hidden Gem
```

This is stronger and more realistic.

---

## 9. Add candidate generation rules

Instead of letting Word2Vec search across all grapes blindly, first create sensible candidate grapes.

Example:

```python
candidate_rules = {
    "tomato_based": [
        "Sangiovese",
        "Barbera",
        "Pinot Noir"
    ],

    "grilled_meat": [
        "Cabernet Sauvignon",
        "Syrah",
        "Malbec"
    ],

    "seafood": [
        "Sauvignon Blanc",
        "Riesling",
        "Pinot Grigio"
    ],

    "spicy": [
        "Riesling",
        "Viognier",
        "Sauvignon Blanc"
    ],

    "creamy": [
        "Chardonnay",
        "Riesling",
        "Sauvignon Blanc"
    ],

    "dessert": [
        "Riesling",
        "Chenin Blanc",
        "Viognier"
    ],

    "fried": [
        "Riesling",
        "Sauvignon Blanc",
        "Pinot Grigio",
        "Chardonnay"
    ]
}
```

This prevents the system from recommending the same few grapes for everything.

---

## 10. Use Word2Vec as ranking support, not the only decision maker

Word2Vec should not be the main judge.

Use Word2Vec to adjust the ranking among already sensible candidates.

Example:

```text
Food profile says pizza should consider:
Sangiovese, Barbera, Pinot Noir

Word2Vec then ranks those candidates based on semantic similarity.
```

This is much safer than:

```text
Word2Vec searches all grapes and keeps picking Barbera or Nebbiolo.
```

---

## 11. Reduce keyword expansion

Your current keyword expansion is too broad.

If a dish starts with 5-6 useful words and expands into 30-50 words, the identity of the dish becomes blurry.

Pizza stops looking like pizza.

Sushi stops looking like sushi.

Everything becomes a generic cloud of:

```text
rich
savory
cooked
fresh
spicy
fruit
mineral
warm
acid
```

### Recommended change

Reduce expansion.

Instead of:

```python
topn = 15
max_total = 50
```

Use:

```python
topn = 5
max_total = 15
```

This keeps the food profile more specific.

---

## 12. Improve the vocabulary filter

Your current `_WINE_VOCAB` is too broad because it includes any word that appears in wine reviews.

But wine reviews can mention food words too.

So words like these can enter the vocabulary:

```text
tomatoes
garlic
onions
mushrooms
bacon
sauce
```

That means the model may translate food words into other food words, not wine descriptors.

### Recommended change

Create an allowed wine descriptor list.

Example:

```python
ALLOWED_WINE_TERMS = {
    "acidic", "crisp", "fresh", "mineral", "citrus", "lemon", "lime",
    "cherry", "raspberry", "blackberry", "plum", "cassis",
    "earthy", "herbal", "spicy", "pepper", "smoky",
    "creamy", "buttery", "oaky", "vanilla",
    "tannic", "dry", "light_body", "medium_body", "full_body",
    "floral", "aromatic", "honey", "stone_fruit"
}
```

Then only allow expansion into these terms.

This prevents bad translations like:

```text
tomato -> garlic, onion, mushrooms
```

and encourages better translations like:

```text
tomato -> cherry, red_fruit, acidity, herbal
```

---

## 13. Redefine Safe Bet, Bold Move, and Hidden Gem

### Safe Bet

Safe Bet should mean:

```text
The most classic and reliable pairing.
```

Example:

```text
pizza -> Sangiovese
steak -> Cabernet Sauvignon
sushi -> Sauvignon Blanc
chicken curry -> Riesling
```

### Bold Move

Bold Move should not mean:

```text
The most mathematically distant grape.
```

That can produce weird results.

Bold Move should mean:

```text
A contrast pairing that solves a food problem.
```

Examples:

```text
fatty food -> high acidity
fried food -> fresh, crisp, mineral
spicy food -> aromatic, low tannin, maybe off-dry
creamy food -> acidity or freshness
sweet dessert -> enough sweetness or aromatic fruit
raw seafood -> saline, mineral, citrus
charred meat -> tannin, dark fruit, smoke, pepper
```

### Hidden Gem

Hidden Gem should not mean:

```text
The next unused grape in the ranking.
```

It should mean:

```text
Less obvious, but still compatible.
```

Example:

```text
pizza:
Safe Bet -> Sangiovese
Bold Move -> Riesling
Hidden Gem -> Barbera

steak:
Safe Bet -> Cabernet Sauvignon
Bold Move -> Riesling or Sauvignon Blanc as a fresh contrast
Hidden Gem -> Syrah

sushi:
Safe Bet -> Sauvignon Blanc
Bold Move -> Riesling
Hidden Gem -> Pinot Grigio
```

---

## 14. Make BiLSTM part of the final score

Currently, the BiLSTM mostly comes after the grape has already been selected.

It retrieves the review, but it does not strongly influence the recommendation.

To make the project stronger, include BiLSTM confidence in the final score.

Example:

```python
final_score = (
    0.45 * rule_based_pairing_score +
    0.25 * word2vec_similarity +
    0.15 * bilstm_confidence +
    0.15 * vivino_rating
)
```

This makes the RNN/BiLSTM branch part of the business recommendation, not just a decoration.

---

## 15. Add a final recommendation score

Each wine card should have a final score.

Example:

```text
Safe Bet: Sangiovese
Bottle: Chianti Classico Riserva 2019
Flavor match: 82%
BiLSTM review confidence: 74%
Vivino approval: 89%
Final recommendation score: 82%
```

This makes the recommendation feel like a real product decision.

---

## 16. Diversify bottle selection

Currently, if the system chooses the same grape, it may also choose the same highest-rated bottle every time.

That makes the app look repetitive.

### Current logic

```python
best = sub.loc[sub["rating_pct"].idxmax()]
```

### Recommended logic

Pick randomly among the top-rated bottles for that grape.

```python
def get_best_wine(grape_name, top_k=10):
    sub = df_wine_mapped[
        df_wine_mapped["grape_class"] == grape_name
    ].dropna(subset=["rating_pct"])

    if sub.empty:
        return "Unknown wine", 0.0

    top = sub.sort_values("rating_pct", ascending=False).head(top_k)
    best = top.sample(1).iloc[0]

    wine_name = str(best["wine_label"])
    rating_pct = round(float(best["rating_pct"]), 1)

    return wine_name, rating_pct
```

This makes the app feel more alive and less repetitive.

---

## 17. Add explanation text to every recommendation

A useful recommender should explain why it recommends something.

Instead of showing only:

```text
Pizza -> Sangiovese
```

Show:

```text
Safe Bet: Sangiovese

Why this works:
Pizza is tomato-based, salty, cheesy, and herbal. Sangiovese works because it has high acidity, red fruit, and herbal notes that match tomato sauce and cut through cheese.
```

Example for sushi:

```text
Safe Bet: Sauvignon Blanc

Why this works:
Sushi is delicate, salty, mineral, and slightly acidic from seasoned rice. Sauvignon Blanc keeps the pairing fresh without overpowering the fish.
```

This makes the project much more useful for a real user and much easier to explain in the presentation.

---

## 18. Add diagnostic checks

After changes, test whether the recommender still collapses.

### Check grape distribution

```python
df_rec["Safe Bet grape"].value_counts()
df_rec["Bold Move grape"].value_counts()
df_rec["Hidden Gem grape"].value_counts()
```

If one grape appears in more than 40-50% of foods, the recommender is still too biased.

### Check recommendations across different foods

Test at least these:

```text
pizza
steak
sushi
hamburger
ramen
pad_thai
chicken_curry
grilled_salmon
caesar_salad
chocolate_cake
apple_pie
oysters
fried_calamari
macaroni_and_cheese
cheese_plate
```

These should not all return the same few grapes.

---

## 19. Improve the 20-example integration table

Your assignment needs examples where both model outputs are displayed side by side.

Your table should include:

```text
Food image/sample
CNN predicted food
CNN confidence
Food profile summary
Safe Bet grape
Bold Move grape
Hidden Gem grape
Bottle name
Review quote
BiLSTM confidence
Vivino rating
Final recommendation score
Explanation
```

Example structure:

| Food | CNN Confidence | Profile | Pairing Type | Grape | Bottle | Review | BiLSTM Conf. | Vivino | Final Score | Why |
|---|---:|---|---|---|---|---|---:|---:|---:|---|
| Pizza | 94% | tomato, cheese, herbs | Safe Bet | Sangiovese | Chianti Classico | “...” | 74% | 89% | 82% | High acidity and red fruit match tomato sauce |

This will make the business integration much clearer.

---

## 20. Recommended final architecture

Use this as the final architecture in the notebook and presentation:

```text
User uploads food photo
        ↓
CNN predicts food label + confidence
        ↓
Food JSON returns structured food profile
        ↓
Rule-based engine creates sensible candidate grapes
        ↓
Word2Vec ranks candidate grapes semantically
        ↓
BiLSTM retrieves real review + confidence for selected grape
        ↓
Vivino table selects bottle/rating
        ↓
Final score ranks Safe Bet, Bold Move, Hidden Gem
        ↓
User receives 3 explainable wine cards
```

This is still an ML project.

The ML components are:

```text
CNN = image classification
Word2Vec = semantic similarity
BiLSTM = review text classification/retrieval
```

The rule-based layer is not a weakness. It is the business integration layer that makes the recommendation safe, explainable, and useful.

---

## 21. Best implementation strategy

Do not rewrite all 101 food entries immediately.

First, fix 10-15 important demo foods:

```text
pizza
steak
sushi
hamburger
ramen
pad_thai
chicken_curry
grilled_salmon
caesar_salad
chocolate_cake
apple_pie
oysters
fried_calamari
macaroni_and_cheese
cheese_plate
```

Make those excellent.

Then use them for:

```text
demo
20-example table
presentation screenshots
prototype testing
```

After that, gradually extend the same structure to all 101 Food-101 classes.

---

## 22. Simple summary

The current project works like this:

```text
Word2Vec guesses wine from poetic food keywords.
```

The improved project should work like this:

```text
CNN recognises food.
Structured food profile describes the dish.
Rule-based engine creates sensible wine candidates.
Word2Vec ranks the candidates.
BiLSTM provides real review evidence.
Final business score produces 3 explainable wine cards.
```

This will make the project more stable, more useful, and easier to defend as an applied deep learning solution.
