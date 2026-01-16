import pandas as pd
import re
from textblob import TextBlob

# -------------------------------------------------
# STEP 1: Load Sentiment140 dataset
# -------------------------------------------------
FILE_NAME = "training.1600000.processed.noemoticon.csv"

df = pd.read_csv(
    FILE_NAME,
    encoding="ISO-8859-1",
    header=None
)

# Assign column names
df.columns = ["Sentiment", "Tweet_ID", "Date", "Query", "User", "Text"]

# Keep only useful columns
df = df[["Date", "Text", "Sentiment"]]

print("Dataset loaded successfully")
print(df["Sentiment"].value_counts())

# -------------------------------------------------
# STEP 2: Sample data (IMPORTANT – avoids freezing)
# -------------------------------------------------
df = df.sample(n=100000, random_state=42)
print("Using sample size:", len(df))

# -------------------------------------------------
# STEP 3: Clean tweet text
# -------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df["Clean_Text"] = df["Text"].apply(clean_text)

# -------------------------------------------------
# STEP 4: TextBlob sentiment analysis
# -------------------------------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Final_Sentiment"] = df["Clean_Text"].apply(get_sentiment)

# -------------------------------------------------
# STEP 5: Save Power BI–ready file
# -------------------------------------------------
OUTPUT_FILE = "twitter_sentiment_ready.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("\nSentiment analysis completed ✅")
print(df["Final_Sentiment"].value_counts())
print(f"\nSaved as: {OUTPUT_FILE}")
