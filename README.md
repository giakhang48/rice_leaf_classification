# Rice Leaf Disease Classification (Classical ML)

Image classification of 8 rice leaf conditions using **only classical machine learning** (no deep learning), built as a course project on the dataset of the *FPTU Can Tho Olympic AI 2026 – Vòng Loại Warmup* competition on Kaggle.

## Results

| Metric | Score |
|---|---|
| Validation Macro F1 (304 held-out images, 8 classes) | **0.957** |
| Validation accuracy | 0.964 |
| Test Macro F1, private score | **0.782** |
| Test Macro F1, public score | 0.702 |

The test scores come from a submission made **after the competition deadline**, so they are not part of the official leaderboard.

Validation is higher than test because the competition test set contains noisy images (occlusion, shadows, colour shift, blur) while validation was split from the clean training data. Augmentation (below) is meant to narrow that gap.

### Model comparison (validation Macro F1)

| Model | Macro F1 |
|---|---|
| Decision Tree | 0.810 |
| KNN | 0.856 |
| Gradient Boosting | 0.887 |
| SVM (RBF) | 0.947 |
| Random Forest | 0.950 |
| **Stacking Ensemble** (RF + SVM + KNN → Logistic Regression) | **0.957** |

The weakest class is *Narrow Brown Leaf Spot* (F1 0.889, recall 0.842), which is also the smallest class (126 training images).

## Pipeline

1. **Split first:** stratified 85/15 train/validation split (1,722 / 304 images) *before* any augmentation, so no augmented copy of a validation image can leak into training.
2. **Preprocessing:** resize to 128×128, then **CLAHE** on the L channel (LAB colour space) to reduce the effect of shadows and uneven lighting.
3. **Hand-crafted features (198 dimensions):**
   - HSV colour histograms (16 bins × 3 channels) plus mean and standard deviation per channel = 54 values
   - **HOG** (9 orientations, 32×32 cells) on the grayscale image to capture lesion shape and edge direction
4. **Augmentation (train set only, OpenCV):** horizontal flip, brightness ±35, rotation ±10°, light Gaussian blur. This turns 1,722 images into 12,054 training samples (7 versions each).
5. **Imbalance handling:** `class_weight="balanced"` where supported, `sample_weight` for Gradient Boosting.
6. **Models:** Decision Tree, KNN, SVM, Random Forest, Gradient Boosting, then a Stacking Ensemble selected by validation Macro F1.

## Repository contents

- `rice_leaf_classification.ipynb`: full notebook (EDA, preprocessing, features, augmentation, model comparison, stacking, evaluation, submission export)
- `requirements.txt`: Python dependencies

## How to run

The dataset belongs to the competition and is **not included** in this repository. To reproduce:

1. Get the data from the competition page on Kaggle (FPTU Can Tho Olympic AI 2026 – Vòng Loại Warmup).
2. Update `BASE_DIR` in the config cell to point to your copy (the notebook defaults to the Kaggle input path).
3. `pip install -r requirements.txt`, then run the notebook top to bottom. Feature extraction on the augmented set takes a while on CPU.

Random seed is fixed (`SEED = 42`) for the split and the models.

## Tools

Python, OpenCV, scikit-image, scikit-learn, pandas, NumPy, Matplotlib
