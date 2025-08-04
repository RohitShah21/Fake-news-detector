# clean_and_visualize.py

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set Seaborn style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv(r"C:\Users\Rohit\OneDrive\Desktop\aitask\FakeNewsNet.csv")

# Clean text function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean 'title' and 'source_domain'
df['title_clean'] = df['title'].apply(clean_text)
df['source_domain_clean'] = df['source_domain'].apply(clean_text)

# Bin tweet_num into categories
df['tweet_bin'] = pd.cut(df['tweet_num'], bins=[-1, 50, 1000000], labels=['low', 'high'])

# Combine cleaned text for modeling
df['text'] = df['title_clean'] + ' ' + df['source_domain_clean'] + ' ' + df['tweet_bin'].astype(str)

# Drop unwanted column
if 'news_url' in df.columns:
    df.drop(columns=['news_url'], inplace=True)

# Limit to 3000 rows and shuffle
df_cleaned = df.sample(frac=1, random_state=42).head(3000)

# Save cleaned dataset
df_cleaned.to_csv(r"C:\Users\Rohit\OneDrive\Desktop\aitask\cleaned_dataset.csv", index=False)
print("✅ Cleaned dataset saved to 'cleaned_dataset.csv' with 3000 rows (news_url removed)")

# ---------- VISUALIZATIONS ---------- #

# 1. Real vs Fake News Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='real')
plt.title("Real vs Fake News Count")
plt.xlabel("Label (1 = Real, 0 = Fake)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Rohit\OneDrive\Desktop\aitask\viz1_real_fake_count.png")

# 2. Tweet Popularity (low/high)
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='tweet_bin')
plt.title("Tweet Popularity Distribution")
plt.xlabel("Tweet Bin")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Rohit\OneDrive\Desktop\aitask\viz2_tweet_bin.png")

# 3. Top 5 Source Domains
plt.figure(figsize=(8, 4))
top_domains = df_cleaned['source_domain_clean'].value_counts().head(5)
sns.barplot(x=top_domains.values, y=top_domains.index)
plt.title("Top 5 Source Domains")
plt.xlabel("Frequency")
plt.ylabel("Source Domain")
plt.tight_layout()
plt.savefig(r"C:\Users\Rohit\OneDrive\Desktop\aitask\viz3_top_domains.png")

# 4. Word Cloud of Titles
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_cleaned['title_clean']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of News Titles")
plt.tight_layout()
plt.savefig(r"C:\Users\Rohit\OneDrive\Desktop\aitask\viz4_wordcloud.png")

# 5. Text Length Distribution
df_cleaned['text_length'] = df_cleaned['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(6, 4))
sns.histplot(df_cleaned['text_length'], bins=30, kde=True)
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words in Text")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(r"C:\Users\Rohit\OneDrive\Desktop\aitask\viz5_text_length.png")

print("📊 Visualizations saved to 'aitask' folder as PNG files.")
