import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n📊 AI Stability Predictor")
print("=====================================")

# 🔹 Ask for file path
file_path = input("📂 Enter Excel file path: ")

# 🔹 Check if file exists
if not os.path.exists(file_path):
    print("❌ File not found. Check path again.")
    exit()

# 🔹 Load data
df = pd.read_excel(file_path)
df.columns = df.columns.str.lower()

print("\nData Preview:")
print(df.head())

# 🔹 Check required columns
required = ['time', 'temperature', 'impurity']
if not all(col in df.columns for col in required):
    print("❌ Your Excel must have columns: TIME, TEMPERATURE, IMPURITY")
    exit()

temps = df['temperature'].unique()

models = {}
results = {}

months = np.array([0, 3, 6, 9, 12, 15, 18, 22]).reshape(-1, 1)

plt.figure()

# 🔹 Train + predict
for temp in temps:
    temp_df = df[df['temperature'] == temp]
    
    model = LinearRegression()
    model.fit(temp_df[['time']], temp_df['impurity'])
    
    predictions = model.predict(months)
    
    models[temp] = model
    results[temp] = predictions
    
    # Plot
    plt.scatter(temp_df['time'], temp_df['impurity'], label=f"{temp}°C Actual")
    plt.plot(months, predictions, label=f"{temp}°C Predicted")

# 🔹 Create table
table = pd.DataFrame({'Months': months.flatten()})

for temp in temps:
    table[f'Impurity_{temp}C'] = results[temp]

print("\n📋 Prediction Table:")
print(table)

# 🔹 Save outputs automatically
output_folder = os.path.dirname(file_path)

table_path = os.path.join(output_folder, "prediction_output.xlsx")
graph_path = os.path.join(output_folder, "stability_graph.png")

table.to_excel(table_path, index=False)
plt.savefig(graph_path)

print(f"\n✅ Table saved at: {table_path}")
print(f"✅ Graph saved at: {graph_path}")

# 🔹 Show graph
plt.xlabel("Time (Months)")
plt.ylabel("Impurity")
plt.title("Impurity vs Time at Different Temperatures")
plt.legend()
plt.show()