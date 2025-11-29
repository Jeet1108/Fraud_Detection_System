# Fraud Detection System - Advanced Models with SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('fraud_data.csv')
X = data.drop('label', axis=1)
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Models
models = {'LogisticRegression': LogisticRegression(max_iter=1000),
          'RandomForest': RandomForestClassifier(),
          'XGBoost': XGBClassifier(eval_metric='logloss'),
          'SVM': SVC(probability=True)}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print(f'\n{name} Performance:')
    print(classification_report(y_test, y_pred))
    # ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:,1]
        print('ROC-AUC:', roc_auc_score(y_test, y_prob))

# Simple dashboard placeholder
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('dashboard_fraud.png')
