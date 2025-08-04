Here's a **README.md** file for your Fake News Detection project, based on the contents of the provided PDF:

---

````markdown
# Fake News Detection System 📰🔍

This repository contains a machine learning-based Fake News Detection System developed using the **Multinomial Naive Bayes** algorithm. The project focuses on detecting misinformation in online news by analyzing article titles. The solution is lightweight, interpretable, and optimized for real-time prediction using a user-friendly **Tkinter GUI**.

## 📁 Project Structure

- `clean.py`: Preprocessing of text (lowercasing, URL & punctuation removal, etc.)
- `train_and_predict.py`: Feature engineering, model training (Naive Bayes), and evaluation
- `gui.py`: Tkinter-based GUI for fake news prediction
- `model.joblib`: Serialized trained model
- `vectorizer.joblib`: Serialized text vectorizer

## 🧠 Key Features

- **Algorithm**: Multinomial Naive Bayes with Laplace smoothing (alpha = 1.0)
- **Dataset**: 3,000 English news headlines from the FakeNewsNet dataset (2016–2019)
- **Feature Engineering**:
  - TF-IDF Vectorization
  - Tweet engagement binning
  - Text length analysis
  - Source domain filtering
- **Evaluation Metrics**:
  - Accuracy: **85.2%**
  - Precision: **87.1%**
  - Recall: **83.4%**
  - F1-score: **85.2%**
  - ROC-AUC, Confusion Matrix
- **Visualizations**:
  - ROC Curve
  - Learning Curve
  - Word Clouds
  - Precision-Recall Curve

## 🎯 Objectives

- Build a scalable and interpretable text classification model
- Provide intuitive real-time prediction via GUI
- Enable reproducibility and extensibility

## 🛠️ Installation

```bash
git clone https://github.com/RohitShah21/Fake-news-detector.git
cd Fake-news-detector
pip install -r requirements.txt
````

## ▶️ Usage

1. Train the model:

   ```bash
   python train_and_predict.py
   ```

2. Launch GUI for predictions:

   ```bash
   python gui.py
   ```

3. Enter a news title and click **"Predict"** to see if it is **Real** or **Fake**.

## 📊 Sample Result (Confusion Matrix)

|                 | Predicted Fake | Predicted Real |
| --------------- | -------------- | -------------- |
| **Actual Fake** | 261 (TN)       | 39 (FP)        |
| **Actual Real** | 50 (FN)        | 250 (TP)       |

## 📈 Strengths

* Lightweight and fast
* High interpretability
* GUI for non-technical users
* Robust to text noise and domain drift

## ⚠️ Limitations

* Only English titles used; full article not analyzed
* Trained on 2016–2019 data (may degrade over time)
* Lacks source credibility or multimedia inputs

## 🚀 Future Enhancements

* Integrate deep learning (e.g., BERT, RoBERTa)
* Add full-article analysis and metadata
* Expand to multilingual detection
* Real-time online learning
* Incorporate social credibility metrics

## 📎 Resources

* 📘 [Project Report (PDF)](./FakeNews.pdf)
* 🧑‍💻 [GitHub Repository](https://github.com/RohitShah21/Fake-news-detector)
* 🎥 [Demo Video](https://github.com/RohitShah21/Fake-news-detector.git) *(YouTube link placeholder)*

## 👤 Author

**Rohit Shah (240228)**
Softwarica College of IT and E-Commerce
Module: STW5000CEM - Introduction to AI
Instructor: Er. Suman Shrestha
Submitted: July 29, 2025

```

---

Let me know if you'd like it saved as a file or converted to DOCX/PDF.
```
