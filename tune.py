import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Paths
base_dir = r"C:\Users\Rohit\OneDrive\Desktop\aitask"
hyperparam_vis_dir = os.path.join(base_dir, "hyperparameter_visualization")
os.makedirs(hyperparam_vis_dir, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(os.path.join(base_dir, "cleaned_dataset.csv"))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text...")
df['title_clean'] = df['title'].fillna('').apply(clean_text)

# Features and target
X = df['title_clean']
y = df['real']

# Vectorize
print("Vectorizing text...")
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning with fewer alphas and no verbose
param_grid = {'alpha': [0.1, 1.0, 2.0]}
print(f"Starting GridSearchCV with params: {param_grid}")

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# Summary of CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
print("\nCV results summary:")
print(cv_results[['param_alpha', 'mean_test_score', 'std_test_score']])

# Plot 1: CV Accuracy vs alpha (mean with std)
plt.figure(figsize=(8,5))
plt.errorbar(
    cv_results['param_alpha'].astype(float), 
    cv_results['mean_test_score'], 
    yerr=cv_results['std_test_score'], 
    fmt='o-', capsize=5
)
plt.xlabel('Alpha')
plt.ylabel('Mean CV Accuracy')
plt.title('Hyperparameter Tuning: Alpha vs Accuracy')
plt.grid(True)
plt.savefig(os.path.join(hyperparam_vis_dir, 'cv_accuracy_vs_alpha.png'))
plt.close()

# Plot 2: Learning curve
print("Generating learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_vec, y, cv=3, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 6)
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, val_mean, label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(hyperparam_vis_dir, 'learning_curve.png'))
plt.close()

# Plot 3: Confusion Matrix on Test Set
print("Evaluating best model on test set...")
y_test_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on Test Set')
plt.savefig(os.path.join(hyperparam_vis_dir, 'confusion_matrix_test.png'))
plt.close()

# Print classification report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

print(f"\n✅ Hyperparameter tuning visualizations saved to '{hyperparam_vis_dir}'")
