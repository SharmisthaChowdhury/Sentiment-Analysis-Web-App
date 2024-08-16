from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('svc_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from request
        statement = data.get('statement', '')  # Extract 'statement'

        # Process the statement and make prediction
        statement_vector = vectorizer.transform([statement])
        prediction = model.predict(statement_vector)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
