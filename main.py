import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open("Base_Model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    location = str(request.form['Location'])
    year = int(request.form['Model Year'])
    kilometers = int(request.form['Kilometers Driven'])
    fuel = str(request.form['Fuel Type'])
    transmission = str(request.form['Transmission'])
    owner = str(request.form['Owner Type'])
    mileage = float(request.form['Mileage'])
    engine = int(request.form['Engine'])
    power = float(request.form['Power'])
    seats = float(request.form['Seats'])

    final_feature = [location, year, kilometers, fuel,
                     transmission, owner, mileage, engine, power, seats]

    user_input = pd.DataFrame(np.array([final_feature]),
                              columns=['Location', 'Year', 'Kilometers_Driven', 'Fuel_Type',
                                       'Transmission', 'Owner_Type', 'Mileage', 'Engine',
                                       'Power', 'Seats'])

    output = model.predict(user_input)

    return render_template("index.html", prediction=f"This car's resale value is approximately {output} lakhs")


if __name__ == '__main__':
    app.run(debug=True)

