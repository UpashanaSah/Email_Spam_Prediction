import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df=pd.read_csv("PhishingEmailData.csv", encoding="latin-1")
# Preprocess the data
df["text"]=df["Email_Subject"].fillna("")+ " " +df["Email_Content"].fillna("")
df["label"]=df["To"].apply (lambda x: 1 if x =="B" else 0)
X=df["text"]
y=df["label"]
# Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Vectorize the text data
vectorizer=TfidfVectorizer()
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)
# Train the model
model=LogisticRegression()
model.fit(X_train_vec,y_train)
# Make predict
predictions=model.predict(X_test_vec)
# Evaluate the model
accuracyresult=accuracy_score(y_test,predictions)
print("Accuracy:", accuracyresult)
print(classification_report(y_test, predictions))

# Save the model and vectorizer
import pickle
pickle.dump(model,open("phishing_model.pkl","wb"))
pickle.dump(vectorizer,open("phishing_vectorizer.pkl","wb"))
