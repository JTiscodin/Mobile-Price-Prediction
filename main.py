import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('Mobile phone price.csv')

data.columns = data.columns.str.strip()

data['Price ($)'] = pd.to_numeric(data['Price ($)'], errors='coerce')
data['RAM'] = data['RAM'].str.extract(r'(\d+)').astype(float)
data['Storage'] = data['Storage'].str.extract(r'(\d+)').astype(float)
data['Screen Size (inches)'] = pd.to_numeric(data['Screen Size (inches)'], errors='coerce')
data['Battery Capacity (mAh)'] = pd.to_numeric(data['Battery Capacity (mAh)'], errors='coerce')

data.dropna(subset=['Price ($)', 'RAM', 'Storage', 'Screen Size (inches)', 'Battery Capacity (mAh)'], inplace=True)

plt.figure(figsize=(10, 6))
plt.hist(data['Price ($)'], bins=30, color='skyblue', edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['RAM'], data['Price ($)'], color='blue', alpha=0.6)
plt.title('Price vs RAM')
plt.xlabel('RAM (GB)')
plt.ylabel('Price ($)')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['Battery Capacity (mAh)'], data['Price ($)'], color='green', alpha=0.6)
plt.title('Price vs Battery Capacity')
plt.xlabel('Battery Capacity (mAh)')
plt.ylabel('Price ($)')
plt.show()

X = data[['RAM', 'Storage', 'Screen Size (inches)', 'Battery Capacity (mAh)']]
y = data['Price ($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

example = {'RAM': 8, 'Storage': 128, 'Screen Size (inches)': 6.5, 'Battery Capacity (mAh)': 4500}
example_df = pd.DataFrame([example])
predicted_price = model.predict(example_df)[0]
print(f"Predicted Price for the example phone: ${predicted_price:.2f}")
