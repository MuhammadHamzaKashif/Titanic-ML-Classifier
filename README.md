# Titanic ML Classifier

Comparing classic scikit-learn classifiers on the Titanic survival dataset. The notebook covers cleaning, feature encoding, scaling, and cross-validated evaluation for KNN, SVM, and logistic regression.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Dataset

Loaded from `seaborn.load_dataset('titanic')`, then cleaned:

- Label and one-hot encoding for categorical features (`sex`, `who`, `embarked`, `class`, and others)
- Median imputation for missing `age`
- Dummy variables for `deck` and `embark_town`

## Models and cross-validated accuracy

| Model                | Accuracy (CV) |
| -------------------- | ------------- |
| K-Nearest Neighbors  | ~97%          |
| SVM (RBF kernel)     | ~99-100%      |
| Logistic Regression  | ~100%         |

## Pipeline

- Encode categorical columns
- Fill missing values in `age`, `embarked`, and `deck`
- Scale features with `StandardScaler`
- Stratified `train_test_split`
- `cross_val_score` for validation
- `classification_report` for precision, recall, and F1

## Stack

- Python, Jupyter, scikit-learn
- pandas, seaborn, NumPy

## Run it

```bash
git clone https://github.com/MuhammadHamzaKashif/Titanic-ML-Classifier.git
cd Titanic-ML-Classifier
pip install pandas seaborn scikit-learn jupyter
jupyter notebook titanic_ml.ipynb
```

## Notes

- Near-perfect scores on Titanic are a classic sign of leakage. The `alive` column is often left in after one-hot encoding and is a direct restatement of the target. Dropping `alive`, `survived`, and any columns derived from them gives a much more honest evaluation (low 80s is typical for this dataset). Worth checking before treating these numbers as a baseline.
