import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
# Load the CSV file
df = pd.read_csv('Data-Melbourne_F.csv')
from sklearn.model_selection import train_test_split
# Drop unnecessary columns
data = df.drop(columns=['year', 'month', 'day'])
data = data.drop(columns=['SLP', 'VV', 'PP', 'VM', 'VG','H','Tm','T','avg_inflow','avg_outflow','V','total_grid','TM'])
# print(data.head(5))

# Identify the features and target variable
features = data.drop(columns=['BOD'])
target = data['BOD']

scaler = StandardScaler()
# Fit and transform the features
scaled_features = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# rf_predictions = rf_model.predict(X_test)
# rf_mae = mean_absolute_error(y_test, rf_predictions)
# rf_r2 = r2_score(y_test, rf_predictions)
# print(f'Random Forest MAE: {rf_mae}')
# print(f'Random Forest R²: {rf_r2}')
dump(rf_model,'water_treatment_regressor.joblib')
