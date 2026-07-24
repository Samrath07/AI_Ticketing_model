
"""
Exploratory Data Analysis (EDA) for the bitext dataset.
Usage:
    python eda_bitext.py --csv bitext_export.csv
"""
import argparse
import re

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"instruction", "category", "intent", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def report_shape_and_balance(df: pd.DataFrame) -> None:
    print("=" * 70)
    print(f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    print("=" * 70)

    print(f"\nCategories: {df['category'].nunique()}")
    print(df["category"].value_counts())

    print(f"\nIntents: {df['intent'].nunique()}")
    print(df["intent"].value_counts())

    print("\nIntents per category (justifies or undercuts a hierarchical classifier):")
    print(df.groupby("category")["intent"].nunique().sort_values(ascending=False))

    print("\nRows per intent, min/max/mean (imbalance check within the flat 27-way problem):")
    counts = df["intent"].value_counts()
    print(f"  min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}  ratio max/min={counts.max()/counts.min():.2f}")


def report_text_length(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TEXT LENGTH (instruction)")
    print("=" * 70)
    df["instr_len"] = df["instruction"].str.split().str.len()
    print(df["instr_len"].describe())
    print("\nBy category:")
    print(df.groupby("category")["instr_len"].mean().sort_values(ascending=False))

    print("\n" + "=" * 70)
    print("TEXT LENGTH (response) -- relevant to retrieval/RAG design")
    print("=" * 70)
    df["resp_len"] = df["response"].str.split().str.len()
    print(df["resp_len"].describe())


def report_placeholder_tokens(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PLACEHOLDER / SLOT TOKENS (e.g. {{Order Number}})")
    print("=" * 70)
    pattern = re.compile(r"\{\{[^}]+\}\}")
    instr_hits = df["instruction"].str.contains(pattern, regex=True)
    resp_hits = df["response"].str.contains(pattern, regex=True)
    print(f"Instructions containing a {{{{slot}}}} token: {instr_hits.sum()} / {len(df)} ({instr_hits.mean():.1%})")
    print(f"Responses containing a {{{{slot}}}} token:    {resp_hits.sum()} / {len(df)} ({resp_hits.mean():.1%})")
    all_tokens = df["instruction"].str.findall(pattern).explode().dropna()
    print("\nMost common slot tokens found:")
    print(all_tokens.value_counts().head(15))


def report_duplicates(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DUPLICATE / TEMPLATED TEXT CHECK")
    print("=" * 70)
    print(f"Unique instructions: {df['instruction'].nunique()} / {len(df)} ({df['instruction'].nunique()/len(df):.1%} unique)")
    print(f"Unique responses:    {df['response'].nunique()} / {len(df)} ({df['response'].nunique()/len(df):.1%} unique)")
    print("\nResponses per intent (fewer unique responses than rows -> consolidate for retrieval):")
    resp_diversity = df.groupby("intent")["response"].nunique()
    rows_per_intent = df.groupby("intent").size()
    print((resp_diversity / rows_per_intent).sort_values().rename("unique_response_ratio"))


def report_lexical_distinctiveness(df: pd.DataFrame, target: str, top_n: int = 10) -> None:
    print("\n" + "=" * 70)
    print(f"LEXICAL DISTINCTIVENESS: chi2(word, {target})")
    print("=" * 70)
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 1))
    X = tfidf.fit_transform(df["instruction"])
    y = df[target]
    chi2_scores, pvals = chi2(X, y)
    terms = tfidf.get_feature_names_out()
    top_idx = chi2_scores.argsort()[::-1][:top_n]
    n_significant = (pvals < 0.05).sum()
    print(f"Words with p < 0.05 against {target}: {n_significant} / {len(terms)}")
    print(f"\nTop {top_n} words by chi2 score:")
    for i in top_idx:
        print(f"  {terms[i]:20s} chi2={chi2_scores[i]:8.2f}  p={pvals[i]:.4f}")


def report_learnability(df: pd.DataFrame, target: str) -> None:
    print("\n" + "=" * 70)
    print(f"LEARNABILITY CHECK: majority baseline vs TF-IDF + LogisticRegression on '{target}'")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        df["instruction"], df[target], test_size=0.2, random_state=42, stratify=df[target]
    )
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train_vec, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test_vec))

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    model_acc = accuracy_score(y_test, model.predict(X_test_vec))

    print(f"  Majority-class baseline : {baseline_acc:.3f}")
    print(f"  TF-IDF + LogReg          : {model_acc:.3f}")
    print(f"  Lift over baseline       : {model_acc - baseline_acc:+.3f}")


# def main():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--csv", required=True)
#     args = parser.parse_args()

#     df = load(args.csv)
#     report_shape_and_balance(df)
#     report_text_length(df)
#     report_placeholder_tokens(df)
#     report_duplicates(df)
#     report_lexical_distinctiveness(df, "category")
#     report_lexical_distinctiveness(df, "intent")
#     report_learnability(df, "category")
#     report_learnability(df, "intent")

