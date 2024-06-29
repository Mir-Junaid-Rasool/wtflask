import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from joblib import dump
# Load the CSV file
df = pd.read_csv('Data-Melbourne_F.csv')
data = df.drop(columns=['year', 'month', 'day'])
data = data.drop(columns=['SLP', 'VV', 'PP', 'VM', 'VG','H','Tm','T','avg_inflow','avg_outflow','V','total_grid','TM'])
# Identify the features and target variable
features = data.drop(columns=['BOD'])
target = data['BOD']
# print(features)

# Define thresholds for BOD categories
def categorize_bod(value):
    if value < 330:
        return 'low'
    elif 330 <= value <= 420:
        return 'medium'
    else:
        return 'high'

# Apply categorization
data['BOD_category'] = data['BOD'].apply(categorize_bod)

# Identify features and target
features = data.drop(columns=['BOD', 'BOD_category'])
target = data['BOD_category']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
dump(rf_classifier,'water_treatment_model.joblib')
dump(scaler,'scaler.joblib')
# Make predictions
# rf_predictions = rf_classifier.predict(X_test)

# # Compute confusion matrix
# cm = confusion_matrix(y_test, rf_predictions)
# print(f'Confusion Matrix:\n{cm}')

# # Calculate accuracy, sensitivity (recall), and specificity
# accuracy = accuracy_score(y_test, rf_predictions)
# sensitivity = recall_score(y_test, rf_predictions, average='weighted')

# # Specificity calculation requires individual true negative rates
# def specificity_score(y_true, y_pred, labels=None):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     tn = cm.sum(axis=1) - cm.diagonal()
#     fp = cm.sum(axis=0) - cm.diagonal()
#     specificity_per_class = tn / (tn + fp)
#     return specificity_per_class.mean()

# specificity = specificity_score(y_test, rf_predictions, labels=['low', 'medium', 'high'])

# print(f'Accuracy: {accuracy}')
# print(f'Sensitivity (Recall): {sensitivity}')
# print(f'Specificity: {specificity}')