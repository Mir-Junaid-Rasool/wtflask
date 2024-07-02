import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('Data-Melbourne_F.csv')
data = df.drop(columns=['year', 'month', 'day'])
data = data.drop(columns=['SLP', 'VV', 'PP', 'VM', 'VG','H','Tm','T','avg_inflow','avg_outflow','V','total_grid','TM'])
# Identify the features and target variable
features = data.drop(columns=['BOD'])
target = data['BOD']
# Assuming `features` and `target` are defined
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(features, target)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
feature_importances_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
plt.title('Feature Importances')

# Save the plot as an image file
plt.savefig('feature_importances.png')
