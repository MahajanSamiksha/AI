import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
hours_studied = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
percentage_scored = np.array([42, 45, 50, 58, 65, 73, 75, 85, 95, 100])

# Create and train the model
model = LinearRegression()
model.fit(hours_studied, percentage_scored)

# Predict prices
predicted_prices = model.predict(hours_studied)

# Plotting
plt.scatter(hours_studied, percentage_scored, color='blue', label='Score')
plt.plot(hours_studied, percentage_scored, color='red', label='Predicted score')
plt.xlabel('Time dedicated (hrs)')
plt.ylabel('Score (%)')
plt.legend()
plt.show()

# Predicting a new house price
new_hours_spent = np.array([[4.5]])
predicted_score = model.predict(new_hours_spent)
print(f"Predicted score with {new_hours_spent[0][0]} hrs spent: {predicted_score[0]:.2f}%")
