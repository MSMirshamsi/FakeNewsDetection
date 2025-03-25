# FakeNewsDetection
FakeNewsSentry: SVM-based fake news detection with NLP and metadata.


![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## Project Description

This project develops a machine learning model to detect fake news using a dataset of news articles with metadata and preprocessed text vectors. The solution leverages Support Vector Machines (SVM) to classify news as "Fake" or "Real," achieving a weighted F1 score of over 80%. The approach includes advanced preprocessing, feature engineering, dimensionality reduction, and model optimization, making it a robust demonstration of machine learning skills for a GitHub portfolio.

### Key Features
- **Data Preprocessing**: Handles missing values and engineers features from metadata (e.g., suspicious indicators for site URLs, types, and authors).
- **Text Processing**: Reduces high-dimensional text vectors (42141 features) to 100 components using TruncatedSVD.
- **Model Architecture**: Uses a two-stage SVM approach:
  1. A text-based SVM predicts probabilities from reduced text vectors.
  2. A final SVM combines metadata features with text probabilities.
- **Visualization**: Includes multiple plots (label distribution, type distribution, text probabilities, TruncatedSVD components).
- **Evaluation**: Optimized for the weighted F1 score with hyperparameter tuning via GridSearchCV.

## Dataset

The dataset is provided in two parts for training and testing:
- **`news_train.csv` and `news_test.csv`**: Metadata including author, publication date, site URL, type, and label (for training).
- **`news_train_text_vectors.npz` and `news_test_text_vectors.npz`**: Sparse matrices of text vectors (1500 training samples, 346 test samples, 42141 features each).

## Requirements

To run this project, install the following Python libraries:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
