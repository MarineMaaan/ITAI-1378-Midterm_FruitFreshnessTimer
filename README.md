# Fruit Freshness Timer

A computer vision powered system to receive an image of a fruit and provide a remaining time the fruit will remain edible (before being mostly spoiled).

## Team Members
* **Benjamin LaCount II** - [W218202670@student.hccs.edu](mailto:W218202670@student.hccs.edu)

---

## Project Tier

**Tier 1:** In addition to image classification, feature extraction will be used for calculating practical information for the user (time to spoilage).

---

## Problem Statement

A significant portion of fresh fruit purchased by US households spoils before it can be consumed, making produce one of the largest categories of household food waste. This spoilage not only represents a direct financial loss for families, totaling billions of dollars annually, but it also contributes to the environmental burden of food in landfills. Consumers often struggle to accurately gauge freshness and predict the shelf-life of their fruit, leading to this cycle of waste.

## Solution Overview

This system will take two user inputs: an **image of a fruit** and its **storage condition** ("Pantry" or "Refrigerator"). A computer vision model will classify the fruit (e.g., "Banana") and its current state, outputting a freshness confidence score (e.g., "80% fresh").

This score is then mapped to a "Total Days" timeline sourced from the **USDA FoodKeeper dataset**. The system calculates the remaining shelf-life (e.g., "2-3 days remaining") and also provides valuable **contextual storage tips** (e.g., "Tip: Skin will blacken in the fridge, but the fruit inside is still good!").

---

## Technical Approach

* **CV Technique:** Image Classification
* **Model:** ResNet50 (pre-trained on ImageNet)
* **Framework:** PyTorch
* **Why this approach:** ResNet50 is a powerful and proven architecture for classification. Using a pre-trained model allows for effective transfer learning, which is ideal for a specialized dataset like fruit freshness, leading to higher accuracy with less training data.

---

## Dataset

### 1. CV Model Training Data
* **Source:** Mendeley Data (FruitVision Dataset)
* **Size:** 81,000+ augmented images
* **Labels:** Fresh, Rotten, and Formalin-mixed (for 5 fruit types: apple, banana, grape, mango, orange)
* **Link:** `https://data.mendeley.com/datasets/xkbjx8959c/2`

### 2. Spoilage Timeline Data
* **Source:** USDA / Data.gov (FoodKeeper Dataset)
* **Size:** ~3,500 entries (filtered to 5 relevant fruits)
* **Features Used:** `Keywords` (to map CV labels), `Pantry_Max`, `Refrigerate_Max`, `..._Metric` (to calculate "Total Days"), and `Refrigerate_tips` (for user advice).
* **Link:** `https://catalog.data.gov/dataset/fsis-foodkeeper-data`

---

## Spoilage Timelines (Ground Truth)

The "days remaining" logic will be based on the following timelines from the USDA FoodKeeper dataset for the fruits in our model. Timelines assume storage on a pantry/counter unless "refrigerated" is specified.

| Fruit | Pantry (Ripe) | Refrigerator (Ripe) |
| :--- | :--- | :--- |
| **Apples** | 3 Weeks | 1-2 Months |
| **Bananas** | 2-5 Days | 2-3 Days (Skin will blacken) |
| **Grapes** | 2-4 Days | 1-2 Weeks |
| **Mangoes** | 3-5 Days | 2-3 Days |
| **Oranges** | 1 Week | 3-4 Weeks |

---

## Success Metrics

* **Primary Metric:** Accuracy (for the 'fresh' vs. 'rotten' classification)
* **Target:** 85-90% accuracy
* **Secondary Metrics:** Inference speed (ms per image)

---

## Week-by-Week Plan

* **Week 10:** Finalize proposal. Acquire and preprocess the **FruitVision dataset** (cleaning, splitting). Acquire and filter the **FoodKeeper dataset** for relevant fruit data.
* **Week 11:** Set up development environment (PyTorch, Google Colab). Begin training the baseline ResNet50 model.
* **Week 12:** Analyze baseline model performance. Experiment with data augmentation and hyperparameter tuning (e.g., learning rate, batch size) to improve accuracy.
* **Week 13:** Finalize the trained classification model. Develop the logic to map the model's confidence score to the **parsed FoodKeeper timelines, keywords, and tips**.
* **Week 14:** Build a simple demo (e.g., Colab notebook or basic web app) to upload an image and display the result. Prepare final presentation slides.
* **Week 15:** Practice and deliver the final presentation.

---

## Resources Needed

* **Compute:** Google Colab
* **Cost:** $0
* **APIs:** None

---

## Risks & Mitigation

| Risk | Probability | Mitigation |
| :--- | :--- | :--- |
| Model confidence does not reliably correlate to spoilage time. | High | Simplify the output. Instead of "3-5 days," the output will be a "freshness" category (e.g., "Peak," "Good," "Use Soon"), which is still more useful than a binary "fresh" or "rotten". |
| Model accuracy is low due to subtle visual differences. | Medium | Use more aggressive data augmentation (color jitter, blur, rotation). If ResNet50 fails, experiment with a more recent architecture like EfficientNetV2. |
| User input ("Pantry" vs. "Fridge") is required. | Low | This is a necessary feature for accuracy. The UI will be designed to make this selection simple and obvious. |

---

## AI Usage Log

* Used generative AI (Gemini) for:
    * Brainstorming and refining the Problem Statement and Solution Overview.
    * Populating the project proposal (technical specs, weekly plan, risks).
    * Analyzing and extracting key features from the FoodKeeper dataset.
    * Writing and updating this README file.

---

## Current Status

- [x] Repository created
- [x] Proposal written
- [ ] Dataset acquired
- [ ] Model training started
- [ ] Demo created
- [ ] Final presentation ready
