# Wine Peer — Project Completion Checklist

> Based on `Final_Project_Advanced_ML.md` requirements. Updated: May 2026.

---

## 5.1 — Data Loading & EDA

### Image Dataset (Food-101)
- [x] Grid of 9–16 sample images with labels (`eda_image_grid.png` — 16 classes)
- [x] Image dimensions and channel statistics (`eda_image_dims.png`, `eda_channel_stats.png`)
- [x] Class distribution bar chart (`eda_image_class_dist.png`)

### Text Dataset (WineSensed)
- [x] Number of samples, text length distribution (histogram)
- [x] Class/target distribution (15 grape varieties, ~85% coverage)
- [x] Word cloud and top-N most frequent terms per class
- [x] 3–5 representative text samples per class

---

## 5.2 — Data Preprocessing

### Image Preprocessing
- [x] Resize to 224×224
- [x] ImageNet mean/std normalisation
- [x] Data augmentation on training set (random flips, rotations, colour jitter) with justification

### Text Preprocessing
- [x] Tokenisation (word-level, justified)
- [x] Vocabulary built on training tokens only; OOV (`<UNK>`) handled
- [x] Padding/truncation to fixed `MAX_SEQ_LEN`; truncation % reported
- [x] GloVe pre-trained embeddings loaded and embedding matrix built

---

## 5.3 — Train / Validation / Test Split

- [x] 70% train / 15% val / 15% test for both datasets
- [x] Fixed random seed for reproducibility
- [x] Stratified by target variable (grape class / food class)
- [x] Test sets frozen until final evaluation

---

## 5.4 — CNN Branch (Image Model)

### CNN from Scratch
- [x] Custom architecture (3+ convolutional blocks with pooling)
- [x] Trainable parameter count reported
- [x] Training and validation loss + accuracy curves (`cnn_scratch_curves.png`)
- [x] Final test accuracy and Macro F1
- [x] Confusion matrix (focused heatmap + worst/best class bars)

### Transfer Learning (ResNet-50)
- [x] Pre-trained backbone loaded (ImageNet weights)
- [x] Two-phase fine-tuning (Phase A: head only; Phase B: head + layer4 unfrozen)
- [x] Training and validation loss + accuracy curves (`resnet50_curves.png`)
- [x] Final test accuracy and Macro F1
- [x] Confusion matrix (focused heatmap + worst/best class bars)

### Comparison
- [x] Grad-CAM explainability visualisations for both models
- [x] Section 9.3b — restore-from-checkpoint cell (rebuild model + reload weights + restore history after disconnect)
- [x] **Written discussion: which model performs better and why** (Section 9.7 — 47.4% scratch vs 78.4% ResNet-50, +31 pp gap explained) ✔

---

## 5.5 — RNN/LSTM Branch (Text Model)

### Architecture
- [x] Embedding layer (GloVe 100-d, fine-tuned) + LSTM/BiLSTM layers
- [x] Variation 1: Unidirectional 2-layer LSTM baseline (`LSTMBaseline`)
- [x] Variation 2: Bidirectional 2-layer LSTM with Bahdanau attention (`BiLSTMAttention`)
- [x] Per-epoch Drive checkpoints for both models (resume-safe after disconnect)
- [x] Training/validation loss + accuracy curves for each variation
- [x] Final test accuracy + Macro F1 + confusion matrix for both
- [x] Side-by-side comparison table + per-class accuracy bar chart

### Bonus (+3 pts)
- [x] Transformer-based encoder (DistilBERT) comparison — Sections 11.10–11.12 ✔

---

## 5.6 — Business Integration

- [ ] Define concrete business decision using both model outputs
- [ ] Implement Word2Vec cosine similarity pairing (characteristic / opposite / unexpected)
- [ ] BiLSTM review retrieval — median-confidence review per grape class (representative, not extreme)
- [ ] Wine ranking — highest-rated wine per grape from df_wine
- [ ] Table of 10–20 examples: CNN prediction + grape recommendations (3 types) + tasting note snippet
- [ ] Comparison chart: CNN accuracy vs BiLSTM accuracy vs pipeline end-to-end

---

## 5.7 — Deployment Prototype

- [ ] User can upload/select an image → CNN prediction + confidence score
- [ ] User can enter text → RNN prediction + confidence score
- [ ] Combined business recommendation displayed
- [ ] Screenshot or screen recording of working prototype included in notebook
- [ ] Hosted link (Hugging Face Spaces / Streamlit Cloud) **or** standalone inference script with instructions

---

## 5.8 — Business Framing

- [ ] What business decision does this pipeline support? Who is the end user?
- [ ] Cost of false positive vs. false negative for each model
- [ ] How would this pipeline integrate into an existing business workflow?
- [ ] Ethical considerations (bias in images/text, fairness across groups)

---

## 6. Bonus — Joint Model (+10 pts)

- [ ] CNN feature vector (before classification head) extracted
- [ ] RNN final hidden state extracted
- [ ] Both vectors concatenated → FC head → single prediction
- [ ] Comparison table: CNN-only vs. RNN-only vs. joint model
- [ ] Dataset has matched image + text per sample with shared label *(Food-101 image + WineSensed review → compatibility label via Word2Vec pairing)*

---

## 7. Deliverables

- [x] Section 12 — Save ALL results: all 5 model weights (CNN scratch, ResNet-50, LSTM, BiLSTM, DistilBERT) + figures synced to Drive
- [ ] Jupyter Notebook (.ipynb) — runs end-to-end without errors, weights saved & loadable
- [ ] PDF export of the notebook — all outputs and plots visible
- [ ] Presentation (.pptx or .pdf) — 15–20 min, all team members' contributions listed
- [ ] Deployment artefact — hosted prototype link or inference script

---

## 8. Presentation Structure

- [ ] Title slide — project title, team, date
- [ ] Business problem & motivation
- [ ] Dataset overview & EDA highlights (2–3 visualisations)
- [ ] Architecture diagrams — CNN, RNN, business integration flowchart
- [ ] Results: CNN — scratch vs. transfer learning, curves, metrics
- [ ] Results: RNN — variations comparison, curves, metrics
- [ ] Business integration — joint recommendations, examples, comparison chart
- [ ] Interpretability — Grad-CAM (done ✓) + attention/SHAP for text
- [ ] Deployment demo — live or screen recording
- [ ] Business impact & ethics
- [ ] Conclusions & lessons learned
- [ ] Team contributions & references

---

## Summary

| Area | Status |
|---|---|
| Image EDA | ✅ Complete |
| Image Preprocessing | ✅ Complete |
| Train/Val/Test Split (both) | ✅ Complete |
| Text EDA & Preprocessing | ✅ Complete |
| CNN from Scratch | ✅ Complete |
| ResNet-50 Transfer Learning | ✅ Complete |
| Grad-CAM Explainability | ✅ Complete |
| CNN vs ResNet written comparison | ✅ Complete (Section 9.7) |
| Section 9.3b — Resume from checkpoint | ✅ Complete |
| LSTM Baseline (11.1–11.5) | ✅ Complete |
| BiLSTM + Attention (11.6–11.9) | ✅ Complete |
| Section 12 — Save all weights/figures | ✅ Complete |
| Business Integration (5.6) | ⬜ Not started |
| Deployment Prototype (5.7) | ⬜ Not started |
| Business Framing (5.8) | ⬜ Not started |
| Bonus Joint Model (+10 pts) | ⬜ Not started |
| PDF export | ⬜ After all cells run |
| Presentation slides | ⬜ Not started |
