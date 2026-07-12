import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
DATA_DIR = "."
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE),
}


def evaluate_dataset(X_train, X_test, y_train, y_test, dataset_name):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rows = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append({
            "Dataset": dataset_name,
            "Algorithm": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        })

    return pd.DataFrame(rows)


def plot_comparison(df, dataset_name, outpath):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    algs = df["Algorithm"].tolist()
    x = np.arange(len(algs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m].values, width, label=m, color=colors[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(algs, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Classification Algorithm Comparison — {dataset_name}")
    ax.legend(loc="lower right", ncol=4, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_spambase():
    spam = pd.read_csv(f"{DATA_DIR}/spambase.csv")
    X = spam.drop(columns=["spam"])
    y = spam["spam"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    results = evaluate_dataset(Xtr, Xte, ytr, yte, "Spambase")
    results.to_csv(f"{RESULTS_DIR}/spambase_results.csv", index=False)
    plot_comparison(results, "Spambase Dataset", f"{FIGURES_DIR}/spambase_comparison.png")
    return results


def run_titanic():
    titanic = pd.read_csv(f"{DATA_DIR}/titanic.csv")
    df = titanic.copy()

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "HasCabin", "FamilySize"]
    X = df[feature_cols]
    y = df["Survived"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    results = evaluate_dataset(Xtr, Xte, ytr, yte, "Titanic")
    results.to_csv(f"{RESULTS_DIR}/titanic_results.csv", index=False)
    plot_comparison(results, "Titanic Dataset", f"{FIGURES_DIR}/titanic_comparison.png")
    return results


if __name__ == "__main__":
    spam_results = run_spambase()
    titanic_results = run_titanic()

    all_results = pd.concat([spam_results, titanic_results], ignore_index=True)
    all_results.to_csv(f"{RESULTS_DIR}/all_results.csv", index=False)

    print(spam_results.to_string(index=False))
    print(titanic_results.to_string(index=False))

    best_spam = spam_results.loc[spam_results["F1 Score"].idxmax()]
    best_titanic = titanic_results.loc[titanic_results["F1 Score"].idxmax()]
    print("Best (Spambase):", best_spam["Algorithm"], best_spam["F1 Score"])
    print("Best (Titanic):", best_titanic["Algorithm"], best_titanic["F1 Score"])