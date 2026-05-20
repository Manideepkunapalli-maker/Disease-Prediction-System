import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("Training.csv")
df = df.drop_duplicates()
df = df.fillna(0)
X = df.drop("prognosis", axis=1)
y = df["prognosis"]
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred)*100,"%")
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(le, open("encoder.pkl", "wb"))
print("Model saved!")