# AI Analytics Challenge 2025 - Financial Fraud Detection

## Problem Statement
**Industry:** Finance – Fraud Detection in Digital Transactions  
**Case Study:** AI-Powered Fraud Detection in Financial Transactions  

Fraud in financial transactions leads to billions of dollars in losses globally. Traditional rule-based fraud detection struggles to keep up with evolving fraudulent techniques. Machine learning models can detect complex patterns in real-time transactions, reducing financial losses, minimizing false positives, and improving security.

### Challenges
- Class imbalance: Fraudulent transactions are rare (~0.1% of all transactions).  
- False positives impact: Too many false alerts frustrate legitimate users.  
- Evolving fraud patterns require adaptive models.  
- Feature engineering to identify new behavioral patterns.  

### Objectives
- Accurately classify fraudulent transactions while minimizing false positives.  
- Optimize precision-recall balance for business-friendly fraud detection.  
- Develop a real-time fraud detection pipeline with minimal latency.  
- Provide interpretable insights for financial analysts.  

---

## Dataset & Model
- **Model Types:**  
  - Supervised: Naïve Bayes, Random Forest, XGBoost  
  - Unsupervised: PCA, K-means, DBScan  
- **Performance Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC, MCC, Jaccard Score, Balanced Accuracy  
- **Feature Importance:** Identifying top features contributing to fraud detection  

---

## Innovation & Approach
- Used Power BI, Pandas, and Seaborn for data exploration and visualization.  
- Preprocessing with Scikit-learn and Pandas for standardization, encoding, and balancing.  
- Introduced a custom feature `isValidTransaction` to flag inconsistencies in transactions.  
- Implemented a Gaussian Naïve Bayes model within an optimized sklearn pipeline for seamless feature transformation and classification.  
- Removed self-transactions where sender and receiver are identical to reduce misleading patterns.  
- Balanced dataset to improve recall and overall fraud detection rates.  

---

## Dashboard & UI
- Data cleaning and preprocessing including handling missing values, normalization, encoding, and balancing.  
- Key Performance Indicators (KPIs): Accuracy, Precision, Recall, F1 Score, ROC AUC Score, Log Loss, Jaccard Coefficient, Matthews Correlation Coefficient (MCC), Balanced Accuracy.  

---

## Analytics & Visualization
- Visualizations include fraud heatmaps, correlation plots, anomaly detection, precision-recall and ROC curves.  
- Best model: Naïve Bayes with balanced test accuracy.  
- Key features influencing fraud detection: Step, Amount, Type.  

---

## Results & Inferences
- High recall (~82%) indicates good detection of fraudulent transactions.  
- ROC AUC of 0.859 shows good distinction between fraud and non-fraud.  
- Precision is low, indicating many false positives; requires further tuning.  
- Balanced accuracy and MCC highlight the need for improved model balance.  

---

## Conclusion & Future Work
- Introduced `isValidTransaction` feature improving detection accuracy.  
- Balanced dataset significantly improved recall.  
- Naïve Bayes chosen for best recall and real-time applicability.  
- Future enhancements: anomaly detection methods, adaptive thresholding, hybrid models, deep learning approaches, GAN for data augmentation.  
- Integration of real-time transaction data and additional features like user behavior and geolocation.  
- Research directions include Graph Neural Networks and blockchain technology.  
- Deployment considerations: cloud deployment, big data handling, edge AI for mobile apps.  

---

This project demonstrates a robust AI-powered financial fraud detection system designed to reduce losses, minimize false positives, and provide actionable insights for financial institutions.
