# Product Review Sentiment Analysis Using Sentence Transformer Embeddings

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/Domain-NLP%20%26%20Sentiment%20Analysis-brightgreen)
![Model](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20PyTorch-red)

### Project Overview
This project develops an end-to-end **Product Review Sentiment Analysis** system that classifies customer feedback into three distinct sentiment categories:
* 🟢 **POSITIVE**
* 🟡 **NEUTRAL**
* 🔴 **NEGATIVE**

By leveraging **Sentence Transformer embeddings**, unstructured textual reviews are transformed into rich 384-dimensional semantic numerical representations. These dense vectors are passed into classical machine learning classifiers to accurately predict sentiment across balanced and imbalanced evaluation metrics.

The system automates feedback interpretation to help businesses monitor product strengths, address negative experiences, and quantify ambiguous customer opinions at scale.

**Notebook Reference:** `Product_Review_Analysis.ipynb`


### Project Objectives

* **Automate Feedback Classification:** Convert unstructured customer text into multi-class sentiment predictions.
* **Semantic Feature Extraction:** Implement transformer-based sentence embeddings (`all-MiniLM-L6-v2`) over traditional TF-IDF approaches.
* **Mitigate Class Imbalance:** Implement and evaluate cost-sensitive/balanced class weighting strategies (`class_weight="balanced"`).
* **Comparative Evaluation:** Benchmark Multiple ML Classifiers using both **Accuracy** and **Macro F1-Score**.
* **Unseen Inference:** Test the optimized pipeline against completely unseen real-world reviews.

### Dataset Description
The model is trained on `Product_Reviews.csv`.

* **Initial Records:** 1,007 rows × 3 columns
* **Unique Entities:** 66 Unique Product IDs, 908 Unique Product Reviews

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Product ID** | Object / String | Unique identifier assigned to a specific product |
| **Product Review** | Object / Text | Free-form customer text response |
| **Sentiment** | Categorical | Target label (`POSITIVE`, `NEUTRAL`, `NEGATIVE`) |

### Technologies & Libraries Used

Language: Python 3.8+
Environment: Google Colab / Jupyter Notebook

Core Libraries:

- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn (ML Classifiers, Model Metrics, Data Splitting)
- pandas, numpy (Data Processing & Matrix Manipulation)
- matplotlib, seaborn (Data Visualizations)
- torch (Transformer Compute Backend)

### Machine Learning Models 

Four distinct classifiers were trained and evaluated on top of the 384-dimensional embeddings.

**Random Forest Classifier**

- Train Accuracy: 100.00% | Train F1: 100.00%
- Test Accuracy: 86.57% | Test F1: 81.80%
- Displays majority class bias toward POSITIVE instances due to unweighted estimators.

**Gradient Boosting Classifier**

- Train Accuracy: 99.88% | Train F1: 99.88%
- Test Accuracy: 84.08% | Test F1: 80.33%
- Slightly lower generalization capability compared to Random Forest; struggles on minority boundaries.

**Logistic Regression (Class-Balanced)**

- Test Accuracy: 73.63% | Weighted F1: 78.00%
- Macro Precision: 51.00% | Macro Recall: 67.00% | Macro F1: ~54.00%

**Class-Level breakdown:**

- NEGATIVE: Precision: 0.39 | Recall: 0.87 | F1: 0.54
- NEUTRAL: Precision: 0.18 | Recall: 0.38 | F1: 0.24
- POSITIVE: Precision: 0.96 | Recall: 0.76 | F1: 0.85

High Negative Recall (87%) makes this configuration strong for capturing customer dissatisfaction.

**Linear Support Vector Machine (Linear SVM)**

- Test Accuracy: 83.08% | Weighted F1: 82.00%
- Macro Precision: 57.39% | Macro Recall: 54.48% | Macro F1: 55.58%

**Class-Level breakdown:**

- NEGATIVE: Precision: 0.47 | Recall: 0.47 | F1: 0.47
- NEUTRAL: Precision: 0.36 | Recall: 0.25 | F1: 0.30
- POSITIVE: Precision: 0.89 | Recall: 0.92 | F1: 0.90

### Key Findings & Limitations
#### Findings

- Dense sentence embeddings eliminate the need for standard manual text preprocessing pipelines (stemming/lemmatization).
- Cost-sensitive weighting (class_weight="balanced") is essential for maintaining minority class performance under an 85% imbalance ratio.
- Linear Support Vector Machines outperform tree-based ensembling when operating directly within high-dimensional vector spaces (384 dimensions).

#### Limitations

- Dataset contains limited samples (N=1005).
- Distinguishing NEUTRAL sentiment remains hard due to semantic overlap with weak POSITIVE/NEGATIVE statements.

### Future Enhancements

- Fine-tuning all-MiniLM-L6-v2 directly via end-to-end PyTorch training loops rather than using it purely as a feature extractor.
- Applying Synthetic Minority Over-sampling (SMOTE) or text augmentation to populate minority classes.
- Deploying the Linear SVM inference model via a lightweight Streamlit web application.
