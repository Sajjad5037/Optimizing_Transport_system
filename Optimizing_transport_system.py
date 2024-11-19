import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox

# Load your data (replace with actual data path)
# Sample data for illustration (Replace with actual data)
data = pd.read_csv('historical_bus_data.csv')

# Feature engineering: Convert time-based data into numerical features
data['Day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek
data['Month'] = pd.to_datetime(data['Date']).dt.month

# Convert Buses_required into categorical classes (Low, Medium, High)
# Quantile-based binning: Divide into 3 equal parts
bins = np.percentile(data['Buses_required'], [0, 33, 66, 100])  # Divide at 33% and 66% percentiles
labels = ['Low', 'Medium', 'High']
data['Buses_required_class'] = pd.cut(data['Buses_required'], bins=bins, labels=labels, include_lowest=True)

# Define your features and target
features = ['Day_of_week', 'Month', 'Previous_demand', 'Holidays', 'Season']
target = 'Buses_required_class'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert the confusion matrix to a DataFrame with labeled rows and columns
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Print model evaluation metrics
print(f"Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{conf_matrix_df}")

# Define the function to predict the number of buses
def predict_buses():
    try:
        # Retrieve input values from the user
        day_of_week = int(day_of_week_entry.get())
        month = int(month_entry.get())
        previous_demand = int(previous_demand_entry.get())
        holiday = int(holiday_entry.get())
        season = int(season_entry.get())
        
        # Create an input array for the model
        features_input = np.array([day_of_week, month, previous_demand, holiday, season]).reshape(1, -1)
        
        # Predict the category of bus demand (Low, Medium, High)
        predicted_class = model.predict(features_input)
        
        # Display the prediction in the result label
        result_label.config(text=f"Predicted Bus Demand Category: {predicted_class[0]}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")

# Create the main window
window = tk.Tk()
window.title("Bus Prediction System")

# Add labels and entry fields for user input
tk.Label(window, text="Day of the Week (0-6)").grid(row=0, column=0)
day_of_week_entry = tk.Entry(window)
day_of_week_entry.grid(row=0, column=1)

tk.Label(window, text="Month (1-12)").grid(row=1, column=0)
month_entry = tk.Entry(window)
month_entry.grid(row=1, column=1)

tk.Label(window, text="Previous Demand").grid(row=2, column=0)
previous_demand_entry = tk.Entry(window)
previous_demand_entry.grid(row=2, column=1)

tk.Label(window, text="Holiday (0 = No, 1 = Yes)").grid(row=3, column=0)
holiday_entry = tk.Entry(window)
holiday_entry.grid(row=3, column=1)

tk.Label(window, text="Season (0 = Off-peak, 1 = Peak)").grid(row=4, column=0)
season_entry = tk.Entry(window)
season_entry.grid(row=4, column=1)

# Add a button to trigger the prediction
predict_button = tk.Button(window, text="Predict Buses", command=predict_buses)
predict_button.grid(row=5, columnspan=2)

# Label to display the prediction result
result_label = tk.Label(window, text="Predicted Bus Demand Category: ", font=("Helvetica", 14))
result_label.grid(row=6, columnspan=2)

# Start the Tkinter event loop
window.mainloop()
