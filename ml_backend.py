from flask import Flask, request, jsonify
import numpy as np
import webbrowser
from threading import Timer
import pickle

model = pickle.load(open('RandomForest.pkl','rb'))
print(1)

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')

    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_query = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    result = model.predict(input_query)[0]

    return jsonify({'crop':str(result)})

if __name__ == '__main__':
    app.run(debug=True)

# def open_browser():
#       webbrowser.open_new("http://127.0.0.1:5000")

# if __name__ == "__main__":
#       Timer(1, open_browser).start()
#       app.run(port=2000)

# flask --app ml_backend run
