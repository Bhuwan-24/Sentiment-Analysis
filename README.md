# 🎬 IMDB Sentiment Analysis using Linear SVM

## 📊 Dataset

* **Name:** IMDB Movie Reviews Dataset
* **Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
* **Size:** 50,000 reviews (balanced: positive & negative)


## Model

* **Algorithm:** Linear Support Vector Machine (SVM)
* **Implementation:** `LinearSVC` (scikit-learn)



## Approach

### 1. Text Vectorization

* Used **TF-IDF Vectorizer** (`max_features = 5000`)
* Converts text into numerical feature vectors
* Output is a **sparse matrix**, which:

  * Reduces memory usage
  * Speeds up training

---

### 2. Data Visualization

* High-dimensional data (5000D) was reduced to **2D using PCA**
* Visualization showed overlapping classes

 Reason:

* PCA reduces dimensions but **loses important information**
* Data may look inseparable in 2D but is separable in higher dimensions



### 3. Model Training

* Used **LinearSVC** because:

  * Optimized for large-scale linear problems
  * Works efficiently with sparse data
  * Faster than `SVC(kernel='linear')`



### Hyperparameter Tuning

* Tuned **C (regularization parameter)** using GridSearchCV
* C controls:

  * Trade-off between margin size and misclassification
* Best value selected based on F1-score



### Evaluation

* Metric: **F1-score**
* Result:

  * **F1-score ≈ 0.86**
  * Balanced performance on both classes



##  Visualization of Hyperplane

* Hyperplane and margins visualized in **3D space**
* Used `SVC(kernel='linear')` for visualization because:

  * Provides easier access to model coefficients (`coef_`)

 Observation:

* Hyperplane may look imperfect in low dimensions
* In high-dimensional space, SVM separates data effectively



## Key Insights

* Sparse representations significantly improve performance
* High-dimensional data behaves differently than low-dimensional projections
* SVM performance is highly sensitive to the **C parameter**
* Visualization in reduced dimensions can be misleading



## Tools & Libraries

* Python
* Scikit-learn
* NumPy
* Pandas
* Matplotlib



## Note

Some visualization concepts were explored with the assistance of AI tools for better understanding and experimentation.



## Future Improvements

* Try **non-linear kernels (RBF)**
* Compare with **Logistic Regression & Naive Bayes**


