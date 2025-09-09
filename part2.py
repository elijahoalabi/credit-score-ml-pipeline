import pandas as pd
import numpy as np
import pickle


from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


le = preprocessing.LabelEncoder()



from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier



Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
    

    saveBestModel(clf)
    print("Best model saved")

df = readData("credit_train.csv")


X = df[Features]
y = df[Label]

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# List of classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Naive Bayes', GaussianNB()),
    ('SVM', SVC(probability=True)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    # Two classifiers of your choice
    ('KNN', KNeighborsClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier())
]


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize results dictionary
results = {}

# Loop over classifiers
for name, clf in classifiers:
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    
    results[name] = {'mean_auc': scores.mean(), 'std_auc': scores.std()}
    print(f"Classifier: {name}, AUROC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hyperparameter tuning
svc_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [0.001, 0.01, 0.1, 1]
}

rf_param_grid = {
    'classifier__n_estimators': [3, 10, 30],
    'classifier__max_depth': [5, 10, 15, 20]
}

# Hyperparameter tuning for SVC
svc_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True))
])

svc_grid_search = GridSearchCV(estimator=svc_pipeline,
                               param_grid=svc_param_grid,
                               cv=cv,
                               scoring='roc_auc',
                               n_jobs=-1)

svc_grid_search.fit(X, y)


svc_best_params = svc_grid_search.best_params_
svc_best_score = svc_grid_search.best_score_
svc_std_score = svc_grid_search.cv_results_['std_test_score'][svc_grid_search.best_index_]

print(f"SVC Best AUROC: {svc_best_score:.4f} (+/- {svc_std_score:.4f}), Best Params: {svc_best_params}")

# Hyperparameter tuning for Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

rf_grid_search = GridSearchCV(estimator=rf_pipeline,
                              param_grid=rf_param_grid,
                              cv=cv,
                              scoring='roc_auc',
                              n_jobs=-1)

rf_grid_search.fit(X, y)


rf_best_params = rf_grid_search.best_params_
rf_best_score = rf_grid_search.best_score_
rf_std_score = rf_grid_search.cv_results_['std_test_score'][rf_grid_search.best_index_]

print(f"Random Forest Best AUROC: {rf_best_score:.4f} (+/- {rf_std_score:.4f}), Best Params: {rf_best_params}")


# Create the pipeline with the best parameters
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(max_depth=5, n_estimators=30))
])

best_pipeline.fit(X, y)

y_pred = best_pipeline.predict(X)
# Generate the confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)

# Calculate precision
precision = precision_score(y, y_pred, pos_label='good')

# Calculate recall
recall = recall_score(y, y_pred, pos_label='good')


y_proba = best_pipeline.predict_proba(X)[:, 1]
auroc = roc_auc_score(y, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUROC: {auroc:.4f}")



output_df = X.copy()
output_df['GroundTruth'] = y
output_df['Prediction'] = y_pred

output_df = output_df[Features + ['GroundTruth', 'Prediction']]


output_df.to_csv('bestModel.output', index=False)


trainOnAllData(df, best_pipeline)


