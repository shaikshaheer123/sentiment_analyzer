from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('logistic_regression.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = model.predict([review])[0]
    return render_template('results.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0', port = 5000)