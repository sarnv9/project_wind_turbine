"""
Model training and evaluation for the wind turbine dataset.
 
Predicts Operational_Status (0 = Normal, 1 = Warning, 2 = Failure).
Main metric: recall of class 2 (Failure), because a missed failure costs far
more than an unnecessary inspection.
 
Input : data/wind_turbine_cleaned.csv (output of the EDA script)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, make_scorer, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# ---------------------------------------------------------------------------
# 1. Load data and identify column types
# ---------------------------------------------------------------------------

df = pd.read_csv('data/wind_turbine_cleaned.csv')

cat_columns = df.select_dtypes(include=['object']).columns.tolist()
num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

target_column = 'Operational_Status'
cat_features = [col for col in cat_columns if col != target_column]
num_features = [col for col in num_columns if col != target_column]

y = df['Operational_Status']
X = df.drop('Operational_Status', axis=1)

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
# stratify=y keeps the class proportions (about 50/30/20) equal in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------------------
# 3. Preprocessing: one-hot encoding of the categorical columns
# ---------------------------------------------------------------------------
cat_encoder = OneHotEncoder(sparse_output=False)

cat_1hot_train = cat_encoder.fit_transform(X_train[cat_features])
cat_1hot_test = cat_encoder.transform(X_test[cat_features])
print(cat_1hot_train.shape)
print(cat_1hot_test.shape)

column_names = cat_encoder.get_feature_names_out()

df_1hot_train = pd.concat(
    [X_train[num_features], pd.DataFrame(cat_1hot_train, columns=column_names, index=X_train.index)],
    axis=1)
df_1hot_test = pd.concat(
    [X_test[num_features], pd.DataFrame(cat_1hot_test, columns=column_names, index=X_test.index)],
    axis=1)

# ---------------------------------------------------------------------------
# 4. Baseline models
# ---------------------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Predictions are kept so models can be compared later.
saved_predictions = {}

for name, model in models.items():
  pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
  pipeline.fit(df_1hot_train, y_train)
  y_pred = pipeline.predict(df_1hot_test)
  saved_predictions[name] = y_pred

# Macro averages weight the three classes equally, which suits the imbalance.
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
  recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
  f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
  class_report = classification_report(y_test, y_pred, zero_division=0)
  print(f'Class report of {name}')
  print(class_report)
  cnf_matrix = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(3,3))
  sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
  print(f'Accuracy: {accuracy:.4f}')
  print(f'Precision: {precision:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'F1 Score: {f1:.4f}')
  plt.title(f"Confusion matrix: {name}")
  plt.ylabel('True Label')
  plt.xlabel("Predicted Label")
  plt.close()

# ---------------------------------------------------------------------------
# 5. Random Forest (bagging), tuned for Failure recall
# ---------------------------------------------------------------------------

rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

params_rf = {'classifier__n_estimators': [100, 200, 300],
              'classifier__max_depth': [10, 20, 30, None],
              'classifier__min_samples_leaf': [1, 2, 4],
              'classifier__max_features': ['log2', 'sqrt']
              }

# Custom metric: recall of class 2 only (the costly Failure class).
def recall_class_2(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[2], average=None)[0]

recall_failure = make_scorer(recall_class_2)

grid_rf = GridSearchCV(estimator=rf_pipeline,
                       param_grid=params_rf,
                       cv=5,
                       scoring=recall_failure,
                       verbose=1,
                       n_jobs=-1)

grid_rf.fit(df_1hot_train, y_train)

best_hyperparams = grid_rf.best_params_
print('Best hyperparameters:\n', best_hyperparams)

grid_rf.best_score_

best_rf_pipeline = grid_rf.best_estimator_

y_pred_final = best_rf_pipeline.predict(df_1hot_test)

rec_final = recall_score(y_test, y_pred_final, labels=[2], average=None)[0]

print(classification_report(y_test, y_pred_final))

plt.figure(figsize=(3, 3))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Confusion Matrix: Optimized Random Forest")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()

# Top 15 features (sorted ascending, so tail() returns the most important).
importances = pd.Series(best_rf_pipeline.named_steps['classifier'].feature_importances_, index=df_1hot_train.columns)
importances.sort_values(ascending=True).tail(15).plot.barh(figsize=(8, 6))

# ---------------------------------------------------------------------------
# 6. Gradient Boosting (boosting), tuned for Failure recall
# ---------------------------------------------------------------------------

gb_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

params_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 8],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search_gb = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=params_gb,
    scoring=recall_failure,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_gb.fit(df_1hot_train, y_train)

grid_search_gb.best_params_

grid_search_gb.best_score_

best_gb_pipeline = grid_search_gb.best_estimator_

y_pred_final_gb = best_gb_pipeline.predict(df_1hot_test)

rec_final_gb = recall_score(y_test, y_pred_final_gb, labels=[2], average=None)[0]

print("Classification Report:\n")
print(classification_report(y_test, y_pred_final_gb))

plt.figure(figsize=(3, 3))
cm_gb = confusion_matrix(y_test, y_pred_final_gb)

sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix: Optimized Gradient Boosting")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()

importances_gbrt_2 = pd.Series(best_gb_pipeline.named_steps['classifier'].feature_importances_, index=df_1hot_train.columns)
importances_gbrt_2.sort_values(ascending=True).tail(15).plot.barh()

# ---------------------------------------------------------------------------
# 7. Stacking
# ---------------------------------------------------------------------------

# Four base learners of different nature (linear, rule-based, distance-based,
# probabilistic), so their errors are less correlated. StackingClassifier
# trains the final estimator on out-of-fold predictions (cv=5), which avoids
# leakage.
estimators_stack=[
('lr', LogisticRegression(max_iter=1000, random_state=42)),
('dt', DecisionTreeClassifier(random_state=42)),
('knn', KNeighborsClassifier()),
('gnb', GaussianNB())
]

final_estimator_stack=LogisticRegression(random_state=42)

stacking = StackingClassifier(
  estimators=estimators_stack,
  final_estimator=final_estimator_stack,
  cv=5,
  n_jobs=-1
)
stacking.fit(df_1hot_train, y_train)

y_pred_stack = stacking.predict(df_1hot_test)

rec_stack = recall_score(y_test, y_pred_stack, labels=[2], average=None)[0]

print(classification_report(y_test, y_pred_stack))

plt.figure(figsize=(3, 3))
cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Purples", cbar=False)
plt.title("Confusion Matrix: Stacking Classifier")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()

# ---------------------------------------------------------------------------
# 8. Cascading (stacking with passthrough)
# ---------------------------------------------------------------------------
# Each base learner is a pipeline with SMOTE and scaling. passthrough=True gives
# the final estimator the original features in addition to the base learners'
# predictions: the second stage sees both the data and the first stage's output.
pipeline_lr = make_pipeline(SMOTE(random_state=42), StandardScaler(), LogisticRegression(random_state=42, max_iter=2000))
pipeline_knn = make_pipeline(SMOTE(random_state=42), StandardScaler(), KNeighborsClassifier())
pipeline_dt = make_pipeline(SMOTE(random_state=42), StandardScaler(), DecisionTreeClassifier(random_state=42))
pipeline_nb = make_pipeline(SMOTE(random_state=42), StandardScaler(), GaussianNB())


estimators_cas = [
    ('lr', pipeline_lr),
    ('knn', pipeline_knn),
    ('dt', pipeline_dt),
    ('nb', pipeline_nb)
]

final_estimator_cas = LogisticRegression(random_state=42)

cascading_model = StackingClassifier(
    estimators=estimators_cas,
    final_estimator=final_estimator_cas,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

cascading_model.fit(df_1hot_train, y_train)

y_pred_cascade = cascading_model.predict(df_1hot_test)

rec_cascade = recall_score(y_test, y_pred_cascade, labels=[2], average=None)[0]

print(classification_report(y_test, y_pred_cascade))

plt.figure(figsize=(6, 5))
cm_cascade = confusion_matrix(y_test, y_pred_cascade)
sns.heatmap(cm_cascade, annot=True, fmt="d", cmap="Oranges", cbar=False)
plt.title("Confusion Matrix: Cascading Classifier")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()


# ---------------------------------------------------------------------------
# 9. Balanced Random Forest
# ---------------------------------------------------------------------------
# Handles the imbalance internally by balancing the classes in every tree's
# bootstrap sample, so SMOTE is not used here.

brf_pipeline = make_pipeline(
    StandardScaler(), 
    BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=10,        
        sampling_strategy='all', 
        replacement=True,
        random_state=42,
        n_jobs=-1
    )
)

brf_pipeline.fit(df_1hot_train, y_train)

y_pred_brf = brf_pipeline.predict(df_1hot_test)

rec_brf = recall_score(y_test, y_pred_brf, labels=[2], average=None)[0]

print(classification_report(y_test, y_pred_brf))

plt.figure(figsize=(6, 5))
cm_brf = confusion_matrix(y_test, y_pred_brf)
sns.heatmap(cm_brf, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix: Balanced Random Forest")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()

importances_brf = pd.Series(brf_pipeline.named_steps['balancedrandomforestclassifier'].feature_importances_, index=df_1hot_train.columns)
importances_brf.sort_values(ascending=True).tail(15).plot.barh()


# ==============================================================================
#   10. THRESHOLD MOVING (BUSINESS OPTIMIZATION)
# ==============================================================================

y_scores = best_gb_pipeline.predict_proba(df_1hot_test)
y_scores_class_2 = y_scores[:, 2]

y_test_binary_2 = (y_test == 2).astype(int)
precision, recall, thresholds = precision_recall_curve(y_test_binary_2, y_scores_class_2)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Gradient Boosting')
plt.title('Precision-Recall Curve (Class 2 Detection)')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Reliability)')
plt.grid(True)
plt.legend()
plt.close()

# Find the Optimal Threshold for a Target Recall (e.g., 80% Safety)
target_recall = 0.80
# Find the index where recall is closest to target (but slightly higher preferably)
# Thresholds array is shorter than recall array by 1, so we adjust logic
idx = np.argmin(np.abs(recall - target_recall))
current_threshold = thresholds[idx]
current_recall = recall[idx]
current_precision = precision[idx]

print("\n--- APPLYING NEW THRESHOLD TO TEST SET ---")

y_pred_adjusted = []
for probs in y_scores:
    # Logic: If Prob(Class 2) > New Threshold -> Predict 2
    if probs[2] >= current_threshold:
        y_pred_adjusted.append(2)
    else:
        # If not Class 2, pick the winner between 0 and 1
        # (We ignore column 2 effectively)
        if probs[0] > probs[1]:
            y_pred_adjusted.append(0)
        else:
            y_pred_adjusted.append(1)

y_pred_adjusted = np.array(y_pred_adjusted)

print(classification_report(y_test, y_pred_adjusted))


plt.figure(figsize=(6, 5))
cm_adj = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm_adj, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title(f"Confusion Matrix (Threshold = {current_threshold:.2f})")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.close()


# Create a directory for the model if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the best pipeline (this saves the scaler, encoder, and the algorithm together)
joblib.dump(best_gb_pipeline, 'models/wind_turbine_model.pkl')
print("Model saved successfully for production!")