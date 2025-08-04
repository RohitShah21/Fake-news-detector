import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)


feat_vis_dir = r"C:\Users\Rohit\OneDrive\Desktop\aitask\featurevisualization"
model_vis_dir = r"C:\Users\Rohit\OneDrive\Desktop\aitask\model visualization"
os.makedirs(feat_vis_dir, exist_ok=True)
os.makedirs(model_vis_dir, exist_ok=True)

df = pd.read_csv(r"C:\Users\Rohit\OneDrive\Desktop\aitask\cleaned_dataset.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['title_clean'] = df['title'].fillna('').apply(clean_text)


df['text_length'] = df['title_clean'].apply(lambda x: len(x.split()))




vectorizer_temp = CountVectorizer(stop_words='english')
X_temp = vectorizer_temp.fit_transform(df['title_clean'])
sum_words = X_temp.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer_temp.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]
words_df = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Frequency', y='Word', data=words_df)
plt.title('Top 20 Most Frequent Words in Titles')
plt.tight_layout()
plt.savefig(os.path.join(feat_vis_dir, "feat1_word_freq.png"))
plt.close()


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='tweet_bin')
plt.title("Tweet Bin Distribution (Low vs High)")
plt.savefig(os.path.join(feat_vis_dir, "feat2_tweet_bin.png"))
plt.close()


plt.figure(figsize=(8, 4))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Title Text Length Distribution")
plt.savefig(os.path.join(feat_vis_dir, "feat3_text_length.png"))
plt.close()



X = df['title_clean']
y = df['real']

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]




plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Fake", "Real"], cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(model_vis_dir, "model1_confusion_matrix.png"))
plt.close()


fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(model_vis_dir, "model2_roc_curve.png"))
plt.close()


precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(os.path.join(model_vis_dir, "model3_precision_recall.png"))
plt.close()


cv_scores = cross_val_score(model, X_vec, y, cv=5, scoring='accuracy')
plt.figure(figsize=(6, 4))
sns.boxplot(cv_scores)
plt.title("Cross-Validation Accuracy Scores")
plt.savefig(os.path.join(model_vis_dir, "model4_cv_accuracy.png"))
plt.close()


feature_names = vectorizer.get_feature_names_out()
log_prob = model.feature_log_prob_[1]  
top20_idx = log_prob.argsort()[-20:]
top_features = [(feature_names[i], log_prob[i]) for i in top20_idx]
feat_df = pd.DataFrame(top_features, columns=["Word", "Log Probability"])

plt.figure(figsize=(10, 5))
sns.barplot(x="Log Probability", y="Word", data=feat_df)
plt.title("Top 20 Predictive Words (Real News)")
plt.savefig(os.path.join(model_vis_dir, "model5_top_words.png"))
plt.close()



print("🔎 Try a new prediction:")
title_input = input("Enter news title: ")
title_cleaned = clean_text(title_input)
vector_input = vectorizer.transform([title_cleaned])
prediction = model.predict(vector_input)
print("✅ Prediction:", "REAL" if prediction[0] == 1 else "FAKE")


print("\n🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
