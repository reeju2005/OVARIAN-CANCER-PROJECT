import numpy as np
from flask import Flask, request, render_template
import pickle

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    pred = None
    error_message = None

    if request.method == 'POST':
        try:
            d1 = float(request.form['a'])
            d2 = float(request.form['b'])
            d3 = float(request.form['c'])
            d4 = float(request.form['d'])
            d5 = float(request.form['e'])
            d6 = float(request.form['f'])
            d7 = float(request.form['g'])
            d8 = float(request.form['h'])
            d9 = float(request.form['i'])
            
            arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9]])
            pred = model.predict(arr)
            print(pred)
        except ValueError:
            error_message = 'Invalid input. Please enter valid numerical values.'

    return render_template('index.html', data=pred, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
