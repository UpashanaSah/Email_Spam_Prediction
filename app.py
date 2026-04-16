from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("phishing_model.pkl","rb"))
vectorizer = pickle.load(open("phishing_vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form["email"]

    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]

    if prediction == 1:
        result = "Phishing Email Detected"
    else:
        result = "Safe Email"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
exit()