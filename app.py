from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('dropout_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['attendance']),
        float(request.form['gpa']),
        float(request.form['engagement']),
        1 if request.form['financial_aid'].lower() == 'yes' else 0,
        1 if request.form['mental_health'].lower() == 'yes' else 0
    ]
    prediction = model.predict([features])[0]
    result = "Student is at risk of dropping out!" if prediction == 1 else "Student is not at risk."
    return render_template('result.html', result=result)

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

if __name__ == '__main__':
    app.run(debug=True)
