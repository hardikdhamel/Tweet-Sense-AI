
import os
import re
import json
import html

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# Configuration
DATASET_NAME = "MrbBakh/Twitter_Laptop"
CSV_FILE = "Twitter_Laptop.csv"
MAX_TWEETS = 3000

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tweets.json")


# Product brands for fallback extraction
BRANDS = [
    "Dell", "ASUS", "Apple", "MacBook", "HP",
    "Lenovo", "Acer", "Microsoft", "Surface",
    "Samsung", "MSI", "Razer", "Toshiba",
    "LG", "Huawei"
]


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_column(columns, candidates):
    """Find a column without case sensitivity."""

    column_map = {
        str(col).lower(): col for col in columns
    }

    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]

    return None


def extract_brand(text, query_value=None):
    """Extract brand from dataset metadata or tweet text."""

    generic_values = {
        "", "nan", "none", "twitterdataset",
        "unknown", "twitter", "dataset"
    }

    if query_value is not None and pd.notna(query_value):
        value = str(query_value).strip()

        if value.lower() not in generic_values:
            return value

    text_lower = text.lower()

    for brand in BRANDS:
        pattern = r"\b" + re.escape(brand.lower()) + r"\b"

        if re.search(pattern, text_lower):
            if brand == "MacBook":
                return "Apple"

            if brand == "Surface":
                return "Microsoft"

            return brand

    return "Unknown"


def load_tweets():

    print("\n========== TWEETSENSE AI ==========")

    # STEP 1: Download dataset
    print("\n[1] Downloading dataset...")

    try:
        file_path = hf_hub_download(
            repo_id=DATASET_NAME,
            filename=CSV_FILE,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN") or None
        )

        print("[SUCCESS] Dataset available!")
        print("File:", file_path)

    except Exception as e:
        print("[ERROR] Download failed:", e)
        return []

    # STEP 2: Read CSV
    print("\n[2] Reading CSV...")

    try:
        df = pd.read_csv(file_path, low_memory=False)

        original_count = len(df)

        print("[SUCCESS] CSV loaded!")
        print("Original records:", original_count)

    except Exception as e:
        print("[ERROR] CSV reading failed:", e)
        return []

    if df.empty:
        print("[ERROR] Dataset is empty!")
        return []

    # STEP 3: Debug dataset structure
    print("\n[3] DATASET DEBUG INFORMATION")

    print("\nAvailable columns:")

    for column in df.columns:
        print(" -", column)

    print("\nFirst record:")
    print(df.iloc[0].to_dict())

    # STEP 4: Detect columns
    print("\n[4] Detecting columns...")

    text_column = find_column(
        df.columns,
        [
            "tweet.full_text",
            "tweet.text",
            "full_text",
            "text"
        ]
    )

    language_column = find_column(
        df.columns,
        [
            "tweet.lang",
            "lang",
            "language"
        ]
    )

    brand_column = find_column(
        df.columns,
        [
            "brand",
            "query_string"
        ]
    )

    date_column = find_column(
        df.columns,
        [
            "tweet.created_at",
            "created_at",
            "date"
        ]
    )

    print("Text column:", text_column)
    print("Language column:", language_column)
    print("Brand/query column:", brand_column)
    print("Date column:", date_column)

    if text_column is None:
        print("[ERROR] Tweet text column not found!")
        return []

    # STEP 5: Clean tweet text
    print("\n[5] Cleaning tweets...")

    df = df.copy()

    df["cleaned_text"] = df[text_column].apply(clean_text)

    print(
        "Non-empty tweets:",
        (df["cleaned_text"] != "").sum()
    )

    df = df[df["cleaned_text"].str.len() >= 15]

    print("After text cleaning:", len(df))

    # STEP 6: Filter English tweets
    if language_column:

        print("\nLanguage distribution:")

        print(
            df[language_column]
            .value_counts()
            .head(10)
        )

        before = len(df)

        df = df[
            df[language_column]
            .astype(str)
            .str.lower()
            .eq("en")
        ]

        print("English tweets:", len(df))
        print("Non-English removed:", before - len(df))

    else:
        print("[WARNING] Language column not found.")

    # STEP 7: Remove duplicates
    before = len(df)

    df = df.drop_duplicates(
        subset=["cleaned_text"]
    )

    print("\nDuplicates removed:", before - len(df))

    if df.empty:
        print("[ERROR] No usable tweets found!")
        return []

    # STEP 8: Select records
    selected_count = min(MAX_TWEETS, len(df))

    df = df.sample(
        n=selected_count,
        random_state=42
    )

    print("\nSelected tweets:", len(df))

    # STEP 9: Prepare final records
    tweets = []

    for _, row in df.iterrows():

        text = row["cleaned_text"]

        query_value = (
            row[brand_column]
            if brand_column
            else None
        )

        brand = extract_brand(
            text,
            query_value
        )

        created_at = (
            str(row[date_column])
            if date_column and pd.notna(row[date_column])
            else ""
        )

        tweets.append({
            "text": text,
            "brand": brand,
            "created_at": created_at
        })

    # STEP 10: Brand distribution
    print("\n[10] BRAND DISTRIBUTION")

    brand_counts = pd.Series(
        [tweet["brand"] for tweet in tweets]
    ).value_counts()

    print(brand_counts)

    # STEP 11: Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tweets,
            file,
            indent=4,
            ensure_ascii=False
        )

    # STEP 12: Final report
    print("\n========== FINAL REPORT ==========")

    print("Original records:", original_count)
    print("Tweets after cleaning:", len(tweets))
    print("Output file:", OUTPUT_FILE)

    print("\nSample tweets:")

    for tweet in tweets[:5]:

        print("\nBrand:", tweet["brand"])
        print("Tweet:", tweet["text"])

    print("\n[SUCCESS] Dataset processing completed!")

    return tweets


if __name__ == "__main__":
    load_tweets()
