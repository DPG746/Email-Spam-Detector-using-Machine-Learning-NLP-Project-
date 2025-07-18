
# 📧 Email Spam Detector using Machine Learning (NLP Project)

This project is a machine learning-based solution that classifies messages as either **spam** or **not spam (ham)** using Natural Language Processing (NLP) techniques. It uses the classic **SMS Spam Collection dataset** and implements a **TF-IDF + Naive Bayes model**.

---

## 🎯 Objective

To build an email (or SMS) classifier that automatically identifies spam messages based on their content using machine learning and text analysis.

---

## 🧠 Key Concepts

- Text Preprocessing
- TF-IDF Vectorization
- Binary Classification
- NLP using Scikit-learn
- Naive Bayes Classifier

---

## 🛠️ Tech Stack

| Category       | Tools/Tech                            |
|----------------|----------------------------------------|
| Programming    | Python 3                               |
| Libraries      | Pandas, NumPy, Matplotlib, Seaborn     |
| NLP Tools      | Scikit-learn (`TfidfVectorizer`, `MultinomialNB`) |
| Deployment     | *(Optional)* Streamlit for simple UI   |

---

## 📂 Dataset

- **Name:** SMS Spam Collection Dataset  
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- **Fields:**
  - `label`: `ham` or `spam`
  - `message`: raw text message content

---

## 📈 Model Workflow

1. Load dataset
2. Clean and preprocess the data
3. Transform text using **TF-IDF**
4. Train using **Multinomial Naive Bayes**
5. Evaluate the model (Accuracy, Confusion Matrix)
6. Save and optionally deploy the model

---

## 🧪 Model Performance

- **Accuracy:** ~95–98% on test set
- **Best Model:** Naive Bayes
- **Evaluation Metrics:**  
  - Accuracy  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score)

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/email-spam-detector.git
cd email-spam-detector
