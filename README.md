# Fruit Freshness Timer

A computer vision powered system to receive an image of a fruit and provide a remaining time the fruit will remain edible (before being mostly spoiled).

## Team Members
* **Benjamin LaCount II** - [W218202670@student.hccs.edu](mailto:W218202670@student.hccs.edu)

---

## Project Tier

**Tier 2:** In addition to image classification, feature extraction will be used for calculating practical information for the user (time to spoilage).

---

## Problem Statement

A significant portion of fresh fruit purchased by US households spoils before it can be consumed, making produce one of the largest categories of household food waste. This spoilage not only represents a direct financial loss for families, totaling billions of dollars annually, but it also contributes to the environmental burden of food in landfills. Consumers often struggle to accurately gauge freshness and predict the shelf-life of their fruit, leading to this cycle of waste.

## Solution Overview

A computer vision model trained on the FruitVision dataset can classify a fruit's current state as 'fresh' or 'spoiled'. The model's confidence in this assessment (e.g., 80% fresh) can then be used as a proxy for its current stage of ripeness. By mapping this confidence score to established spoilage timelines for that specific fruit, the system can translate the 'freshness' percentage into an estimated number of remaining edible days.

---

## Technical Approach

* **CV Technique:** Image Classification
* **Model:** ResNet50 (pre-trained on ImageNet)
* **Framework:** PyTorch
* **Why this approach:** ResNet50 is a powerful and proven architecture for classification. Using a pre-trained model allows for effective transfer learning, which is ideal for a specialized dataset like fruit freshness, leading to higher accuracy with less training data.

---

## Dataset

* **Source:** Mendeley Data (FruitVision Dataset)
* **Size:** 81,000+ augmented images
* **Labels:** Fresh, Rotten, and Formalin-mixed (for 5 fruit types: apple, banana, grape, mango, orange)
* **Link:** `https://data.mendeley.com/datasets/xkbjx8959c/2`

---

## Success Metrics

* **Primary Metric:** Accuracy (for the 'fresh' vs. 'rotten' classification)
* **Target:** 85-90% accuracy
* **Secondary Metrics:** Inference speed (ms per image)

---

## Week-by-Week Plan

* **Week 10:** Finalize proposal, acquire and preprocess the FruitVision dataset (cleaning, splitting into train/validation/test sets).
* **Week 11:** Set up development environment (PyTorch, Google Colab). Begin training the baseline ResNet50 model.
* **Week 12:** Analyze baseline model performance. Experiment with data augmentation and hyperparameter tuning (e.g., learning rate, batch size) to improve accuracy.
* **Week 13:** Finalize the trained classification model. Develop the logic to map the model's confidence score to an estimated "days to spoilage" timeline.
* **Week 14:** Build a simple demo (e.g., Colab notebook or basic web app) to upload an image and display the result. Prepare final presentation slides.
* **Week 15:** Practice and deliver the final presentation.

---

## Resources Needed

* **Compute:** Google Colab (with T4 GPU)
* **Cost:** $0
* **APIs:** None

---

## Risks & Mitigation

| Risk | Probability | Mitigation |
| :--- | :--- | :--- |
| Model confidence does not reliably correlate to spoilage time. | High | Simplify the output. Instead of "3-5 days," the output will be a "freshness" category (e.g., "Peak," "Good," "Use Soon"), which is still more useful than a binary "fresh" or "rotten". |
| Model accuracy is low due to subtle visual differences. | Medium | Use more aggressive data augmentation (color jitter, blur, rotation). If ResNet50 fails, experiment with a more recent architecture like EfficientNetV2. |

---

## AI Usage Log

* Used generative AI to assist with code generation.

---

## Current Status

- [x] Repository created
- [x] Proposal written
- [ ] Dataset acquired
- [ ] Model training started
- [ ] Demo created
- [ ] Final presentation ready
