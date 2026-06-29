# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Remove unnecessary column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Model evaluation
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Predict future sales
new_data = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [50]
})

prediction = model.predict(new_data)

print("\nPredicted Sales:", prediction[0])

# -----------------------------
# Data Visualization
# -----------------------------

# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Actual vs Predicted Sales
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:")
print(importance)

# Feature Importance Graph
plt.figure(figsize=(6, 4))
plt.bar(importance['Feature'], importance['Coefficient'])
plt.xlabel("Advertising Platform")
plt.ylabel("Impact on Sales")
plt.title("Feature Importance")
plt.show()