import argparse
import re
from pathlib import Path
from dataset_eda import load, report_shape_and_balance, report_text_length, report_placeholder_tokens, report_duplicates, report_lexical_distinctiveness, report_learnability
from train_ticket_classifier import check_class_balance, train_category_model, train_intent_models_per_category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="artifacts")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    df = load(args.csv)
    report_shape_and_balance(df)
    report_text_length(df)
    report_placeholder_tokens(df)
    report_duplicates(df)
    report_lexical_distinctiveness(df, "category")
    report_lexical_distinctiveness(df, "intent")
    report_learnability(df, "category")
    report_learnability(df, "intent")
    check_class_balance(df)
    train_category_model(df, output_dir)
    train_intent_models_per_category(df, output_dir)
    
if __name__ == "__main__":
    main()