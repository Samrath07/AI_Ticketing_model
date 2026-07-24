
"""
Usage:
    python train_classifier_bitext.py --csv bitext_export.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

TEXT_COL = "instruction"
CATEGORY_COL = "category"
INTENT_COL = "intent"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {TEXT_COL, CATEGORY_COL, INTENT_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing from CSV: {missing}")
    return df


def check_class_balance(df: pd.DataFrame) -> None:
    print("Category distribution:")
    print(df[CATEGORY_COL].value_counts())
    print("\nIntents per category:")
    print(df.groupby(CATEGORY_COL)[INTENT_COL].nunique())


def train_category_model(df: pd.DataFrame, output_dir: Path, random_state: int = 42) -> TfidfVectorizer:
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[CATEGORY_COL], test_size=0.2, random_state=random_state, stratify=df[CATEGORY_COL]
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=random_state)
    model.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    print(f"[category] accuracy = {acc:.3f}")

    joblib.dump(vectorizer, output_dir / "category_vectorizer.pkl")
    joblib.dump(model, output_dir / "category_model.pkl")
    with open(output_dir / "category_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)
    return vectorizer


def train_intent_models_per_category(df: pd.DataFrame, output_dir: Path, random_state: int = 42) -> None:
    intent_artifacts = {}
    metrics = {}
    for category, group in df.groupby(CATEGORY_COL):
        n_classes = group[INTENT_COL].nunique()
        if n_classes < 2:
            intent_artifacts[category] = {"single_class": group[INTENT_COL].iloc[0]}
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            group[TEXT_COL], group[INTENT_COL], test_size=0.2, random_state=random_state,
            stratify=group[INTENT_COL] if group[INTENT_COL].value_counts().min() >= 2 else None,
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2), min_df=1)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=random_state)
        model.fit(X_train_vec, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_vec))
        metrics[category] = acc
        print(f"[intent | {category}] accuracy = {acc:.3f}  ({n_classes} intents, n={len(group)})")
        print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))

        intent_artifacts[category] = {"vectorizer": vectorizer, "model": model}

    joblib.dump(intent_artifacts, output_dir / "intent_models_by_category.pkl")
    with open(output_dir / "intent_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def predict(text: str, output_dir: str = "artifacts") -> dict:
    out_dir = Path(output_dir)
    cat_vectorizer = joblib.load(out_dir / "category_vectorizer.pkl")
    cat_model = joblib.load(out_dir / "category_model.pkl")
    intent_artifacts = joblib.load(out_dir / "intent_models_by_category.pkl")

    category = cat_model.predict(cat_vectorizer.transform([text]))[0]
    artifact = intent_artifacts[category]

    if "single_class" in artifact:
        intent = artifact["single_class"]
    else:
        intent = artifact["model"].predict(artifact["vectorizer"].transform([text]))[0]

    return {"category": category, "intent": intent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    df = load_data(args.csv)
    check_class_balance(df)
    train_category_model(df, output_dir)
    train_intent_models_per_category(df, output_dir)


if __name__ == "__main__":
    main()