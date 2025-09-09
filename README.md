# Credit Score Classification - Machine Learning Pipeline

This project implements a machine learning pipeline to classify individuals as having **good** or **bad** credit scores using 19 hidden features. 

---


- **Part 1:**  
  - Visualization of decision boundaries across multiple datasets.  
  - Comparison of Logistic Regression, Naive Bayes, SVM, Decision Tree, and Random Forest.  

- **Part 2:**  
  - Machine learning pipeline applied to the credit dataset (`credit_train.csv`).  
  - k-fold cross-validation (k=10) with AUROC evaluation.  
  - Models implemented:  
    - Logistic Regression  
    - Naive Bayes  
    - SVM (SVC)  
    - Decision Tree  
    - Random Forest  
    - K-Nearest Neighbors (KNN)  
    - Gradient Boosting  
  - Hyperparameter tuning with GridSearchCV for SVC and Random Forest.  
  - Model evaluation using confusion matrix, accuracy, precision, recall, and AUROC.  
  - Saving of the best-performing model (`bestModel.model`) and its outputs (`bestModel.output`).  

---

## Results Summary
- **Best Model (before tuning):** Random Forest (Mean AUROC = 0.9230, Std = 0.0388).  
- **Best Model (after tuning):** Random Forest with `max_depth=5`, `n_estimators=30` (Mean AUROC = 0.9339, Std = 0.0322).  
- **Final Performance:** 92.6% accuracy on classification task.  


