import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from joblib import dump
import warnings

warnings.filterwarnings("ignore")

# Load the CSV file
df = pd.read_csv('Data-Melbourne_F.csv')

# Drop irrelevant columns
data = df.drop(columns=['year', 'month', 'day', 'SLP', 'VV', 'VM', 'VG', 'Tm', 'total_grid', 'TM'])
# Handle missing values
data.fillna(data.median(), inplace=True)

# Calculate the thresholds for COD categories
low_threshold = df['COD'].quantile(0.33)
high_threshold = df['COD'].quantile(0.67)

# Define thresholds for COD categories
def categorize_cod(value):
    if value < low_threshold:
        return 'low'
    elif low_threshold <= value <= high_threshold:
        return 'medium'
    else:
        return 'high'
    
# Apply categorization
data['COD_category'] = data['COD'].apply(categorize_cod)

# Identify features and target
features = data[['BOD', 'TN', 'Am', 'avg_inflow', 'avg_outflow', 'T', 'H', 'PP']]
target = data['COD_category']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier model
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model
rf_classifier.fit(X_train, y_train)

# Save the model and scaler
dump(rf_classifier, 'cod_prediction_model.joblib')
dump(scaler, 'codscaler.joblib')

# Make predictions
rf_predictions = rf_classifier.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, rf_predictions)
print(f'Confusion Matrix:\n{cm}')

# Calculate accuracy, sensitivity (recall), and specificity
accuracy = accuracy_score(y_test, rf_predictions)
sensitivity = recall_score(y_test, rf_predictions, average='weighted')

# Specificity calculation requires individual true negative rates
def specificity_score(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum(axis=1) - cm.diagonal()
    fp = cm.sum(axis=0) - cm.diagonal()
    specificity_per_class = tn / (tn + fp)
    return specificity_per_class.mean()

specificity = specificity_score(y_test, rf_predictions, labels=['low', 'medium', 'high'])

print(f'Accuracy: {accuracy}')
print(f'Sensitivity (Recall): {sensitivity}')
print(f'Specificity: {specificity}')
