import pickle
import pandas as pd
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
df = pd.read_csv("Training.csv")
columns = list(df.drop("prognosis", axis=1).columns)
print("\n=== Disease Prediction System ===")
print("\nAvailable symptoms:")
for i, col in enumerate(columns):
    print(f"{i}: {col}")
selected = input("\nEnter symptoms (comma separated, name or index):\n")
input_data = [0] * len(columns)
for item in selected.split(","):
    item = item.strip()
    if item.isdigit():
        idx = int(item)
        if 0 <= idx < len(columns):
            input_data[idx] = 1
    else:
        if item in columns:
            idx = columns.index(item)
            input_data[idx] = 1
input_df = pd.DataFrame([input_data], columns=columns)
prediction = model.predict(input_df)
result = le.inverse_transform(prediction)[0]
print("\nPredicted Disease:", result)