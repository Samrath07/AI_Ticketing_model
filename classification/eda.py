import argparse
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TEXT_COL = "Ticket Description"
TARGETS = ["Ticket Type", "Ticket Priority"]
METADATA_COLS = ["Ticket Channel", "Product Purchased", "Customer Gender", "Ticket Subject"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["clean_desc"] = df[TEXT_COL].str.replace(r"\{product_purchased\}", "", regex=True)
    return df


def report_class_balance(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("CLASS BALANCE")
    print("=" * 70)
    for target in TARGETS:
        print(f"\n{target}:")
        print(df[target].value_counts(normalize=True).round(3))


def report_metadata_independence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("METADATA vs TARGET INDEPENDENCE (chi-square test)")
    print("p > 0.05 means no statistically significant relationship")
    print("=" * 70)
    for target in TARGETS:
        print(f"\n{target}:")
        for col in METADATA_COLS + [t for t in TARGETS if t != target]:
            ct = pd.crosstab(df[col], df[target])
            chi2, p, _, _ = chi2_contingency(ct)
            flag = "SIGNIFICANT" if p < 0.05 else "no signal"
            print(f"  {col:20s} chi2={chi2:8.2f}  p={p:.4f}  [{flag}]")


def report_text_learnability(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TEXT LEARNABILITY: majority baseline vs TF-IDF + LogisticRegression")
    print("=" * 70)
    for target in TARGETS:
        X = df["clean_desc"]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        baseline = DummyClassifier(strategy="most_frequent").fit(X_train_tfidf, y_train)
        baseline_acc = accuracy_score(y_test, baseline.predict(X_test_tfidf))

        model = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
        model_acc = accuracy_score(y_test, model.predict(X_test_tfidf))

        lift = model_acc - baseline_acc
        verdict = "learnable signal found" if lift > 0.05 else "NO meaningful signal above baseline"
        print(f"\n{target}:")
        print(f"  Majority-class baseline : {baseline_acc:.3f}")
        print(f"  TF-IDF + LogReg          : {model_acc:.3f}")
        print(f"  Lift over baseline       : {lift:+.3f}  -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to customer_support_tickets.csv")
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    report_class_balance(df)
    report_metadata_independence(df)
    report_text_learnability(df)


if __name__ == "__main__":
    main()