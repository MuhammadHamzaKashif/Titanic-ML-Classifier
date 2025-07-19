# Titanic-ML-Classifier
A machine learning classification task based on the Titanic dataset using different models like KNN, SVM and Logistic Regression

---

## Dataset

- Source: `seaborn.load_dataset('titanic')`
- Cleaned and preprocessed using:
  - Label encoding & one-hot encoding for categorical features
  - Median imputation for missing age values
  - Added dummy variables for `deck` and `embark_town`

---

## Models Used

| Model                | Accuracy (Cross-Validation) |
|---------------------|-----------------------------|
| K-Nearest Neighbors |  ~97%                      |
| SVM (RBF Kernel)     |  ~99%–100%                |
| Logistic Regression |  ~100%                     |

---

## Preprocessing Steps

- Encoded categorical columns (`sex`, `who`, `embarked`, `class`, etc.)
- Filled missing values in `age`, `embarked`, and `deck`
- Used `StandardScaler` for feature scaling
- Splitting data using `train_test_split` with `stratify` for balanced classes

---

## ML Techniques

- Supervised classification
- Cross-validation with `cross_val_score`
- Evaluation using `classification_report` (Precision, Recall, F1)

---

## Tech Stack

- Python 
- Jupyter Notebook 
- scikit-learn 
- Pandas, Seaborn, NumPy

---

## How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/Titanic-ML-Classifier.git
   ```
2. Run the notebook using Jupyter
   ```bash
    jupyter notebook
   ```
   
---

##Sample Output

              precision    recall  f1-score   support
           0       1.00      1.00      1.00       110
           1       1.00      1.00      1.00        69
    accuracy                           1.00       179
   macro avg       1.00      1.00      1.00       179
weighted avg       1.00      1.00      1.00       179

