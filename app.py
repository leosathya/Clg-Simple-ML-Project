import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
diabetis_model = pickle.load(open('diabetis_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetis')
def diabetis():
    return render_template('diabetis.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/diabetisPredict',methods=['POST'])
def diabetisPredict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = diabetis_model.predict(final_features)

    #output = round(prediction[0], 2)

    return render_template('result.html', data=prediction)

@app.route('/heartPredict',methods=['POST'])
def heartPredict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = heart_model.predict(final_features)

    #output = round(prediction[0], 2)

    return render_template('result2.html', data=prediction)



if __name__ == "__main__":
    app.run(debug=True)