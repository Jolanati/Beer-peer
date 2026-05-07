---
title: Wine & Dine
emoji: 🍷
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: "5.6.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

## 🍽️ Wine & Dine

**Upload a food photo — we identify it, analyze its flavor profile in real time, and recommend your perfect wine.**

## How it works

| Step | What happens |
| --- | --- |
| **1. Upload photo** | Any food photo — from your phone, camera, or the web |
| **2. CNN identifies food** | ResNet-50 (fine-tuned on Food-101) predicts the food class with confidence score and top-5 alternatives |
| **3. Confirm** | You confirm the prediction (or try a different photo) |
| **4. BiLSTM encodes flavor** | The food's flavor description is tokenized and encoded live through a trained BiLSTM with Bahdanau attention — attention weights are visualized word by word |
| **5. Cluster match** | The 512-d BiLSTM vector is compared to 9 flavor cluster centroids (BisectingKMeans) via cosine similarity |
| **6. Wine card** | Three wine pairings: 🥇 Safe Bet · ✨ Characteristic · 🔄 Contrast |

## Models

| Model | Task | Architecture |
| --- | --- | --- |
| **ResNet-50** | Food-101 classification (101 classes) | Fine-tuned on Food-101, two-phase training |
| **BiLSTMAttention** | Flavor encoding | 2-layer bidirectional LSTM + Bahdanau attention, GloVe 100-d embeddings |
| **BisectingKMeans** | Flavor clustering | K=9 clusters on BiLSTM taste vectors |

## Project

RSU Advanced Machine Learning — Final Group Project, 2026  
Branch: `W2V-expansion-layer` · Repo: `Jolanati/wine-dine`

## Files required in Space

```text
app.py
requirements.txt
weights/
  cnn_resnet50_best.pt
  bilstm_best.pt
data/
  vocab.json
  cluster_names.json
  centroids.npy
  results_all.json
  food_flavor_description_v2.json
```
