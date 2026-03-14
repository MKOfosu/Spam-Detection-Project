# SMS Spam Detection

A machine learning project that classifies SMS messages as spam or ham (legitimate) using Natural Language Processing (NLP) techniques and a systematic comparison of vectorization strategies, classifiers, hyperparameter tuning, dimensionality reduction, and resampling methods.

---

## Project Overview

This project applies NLP and supervised learning to the SMS Spam Collection dataset. It goes beyond basic model training to explore a comprehensive set of experiments: six classifiers, two vectorization approaches, GridSearchCV hyperparameter tuning, PCA-based dimensionality reduction, and both downsampling and upsampling strategies to address class imbalance.

---

## Dataset

- **Source:** [SMS Spam Collection](https://raw.githubusercontent.com/BuhariS/spam_sms_collection/refs/heads/main/sms_spam_collection.csv)
- **Size:** 5,572 SMS messages
- **Class Distribution:**
  - Ham (legitimate): 4,825 messages (86.6%)
  - Spam: 747 messages (13.4%)

> Note: The dataset is class-imbalanced, with spam representing roughly 1 in 7 samples. This motivated the resampling experiments later in the project.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| NLP | NLTK (stopwords), scikit-learn |
| Modeling | scikit-learn |
| Environment | Jupyter Notebook |

---

## Workflow

### 1. Data Loading and Exploration
- Loaded dataset directly from GitHub
- Inspected class distribution with value counts and bar chart visualization

### 2. Preprocessing
- Converted labels to binary: `ham → 0`, `spam → 1`
- Lowercased all messages
- Removed punctuation using a custom function

### 3. Feature Engineering
Two vectorization strategies were compared across all models:

- **CountVectorizer** — bag-of-words with bigrams, top 2,500 features, English stopword removal
- **TF-IDF Vectorizer** — term frequency-inverse document frequency with the same configuration

Both used `ngram_range=(1, 2)` and `max_features=2500`.

### 4. Train/Test Split
- 80% training / 20% test split (`random_state=42`)

### 5. Multi-Model Comparison
A reusable `model_performance()` helper function was built to train and evaluate all models systematically across accuracy, recall, precision, and F1-score. Six classifiers were evaluated on both vectorized representations:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Classifier (SVC, RBF kernel)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

### 6. Hyperparameter Tuning with GridSearchCV
GridSearchCV with StratifiedKFold (5 folds) was applied to the two strongest candidates, optimizing for F1-score to account for class imbalance:

- **Random Forest** — tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight`
- **Logistic Regression** — tuned over regularization strength `C` and penalty type (`l1`, `l2`, `elasticnet`)

### 7. Dimensionality Reduction with PCA
PCA was applied to reduce the TF-IDF feature space from 2,500 to 250 components (10% of original features) after StandardScaler normalization. The 250 components captured a substantial proportion of the variance in the training set. Logistic Regression and Random Forest were both retrained on the PCA-reduced features.

**Key finding:** Logistic Regression on PCA-reduced features outperformed all previous model configurations, including the GridSearchCV-tuned models, using only 10% of the original features. Random Forest underperformed Logistic Regression on the reduced space.

### 8. Resampling Experiments
To further address class imbalance, two resampling strategies were applied to the PCA-transformed training data:

- **Downsampling** — majority class (ham) reduced to match the minority class (spam) size
- **Upsampling** — minority class (spam) oversampled with replacement to match ham

**Key finding:** Both resampling techniques worsened model performance compared to the PCA + Logistic Regression baseline, suggesting the model was already handling the imbalance adequately without resampling.

### 9. Inference on Fresh Data
The final model was tested on unseen messages:

```
Message: "Frank you have just won 3 million, click the link to claim it"
Prediction: Spam

Message: "I love you so much"
Prediction: Ham
```

---

## Key Results

| Configuration | Notes |
|---|---|
| Logistic Regression + CountVectorizer | Baseline |
| Logistic Regression + TF-IDF | Marginal improvement over CountVectorizer |
| Random Forest + CountVectorizer / TF-IDF | Competitive but slower |
| GridSearchCV (RF + LR) | Incremental gains over defaults |
| **PCA (250 components) + Logistic Regression** | **Best overall performance** |
| PCA + Random Forest | Underperformed vs. LR on reduced space |
| Downsampling / Upsampling + LR | Degraded performance vs. PCA baseline |

The best result was achieved with PCA-reduced TF-IDF features and Logistic Regression, demonstrating that aggressive dimensionality reduction can improve generalization even when it discards 90% of features.

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/MKOfosu/Spam-Detection-Project.git
   cd Spam-Detection-Project
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```

3. **Download NLTK stopwords**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook "Spam Detection Project.ipynb"
   ```

---

## Repository Structure

```
Spam-Detection-Project/
│
├── Spam Detection Project.ipynb   # Main analysis notebook
└── README.md                      # Project documentation
```

---

## Future Improvements

- Explore SMOTE for synthetic minority oversampling instead of random resampling
- Test transformer-based embeddings (e.g., sentence-transformers) as an alternative to TF-IDF
- Vary the number of PCA components systematically to find the optimal variance-performance tradeoff
- Deploy the best model as a lightweight web API for real-time spam classification

---

## Author

**Mathias Ofosu** — Data Scientist
LinkedIn: [linkedin.com/in/mathias-ofosu](https://www.linkedin.com/in/mathias-ofosu/)  
GitHub: [github.com/MKOfosu](https://github.com/MKOfosu/)  
X: [@MKOfosu](https://x.com/MKOfosu)

---

## License

This project is open source and available under the [MIT License](LICENSE).