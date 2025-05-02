from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        result = "Positive 😊" if prediction == 1 else "Negative 😞"
        return render_template('result.html', text=text, result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)