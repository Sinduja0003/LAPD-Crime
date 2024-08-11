from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import pickle

scaler = StandardScaler()

# models eekada
with open('./models/area.pkl', 'rb') as f:
    areaModel = pickle.load(f)
with open('./models/CrimeType.pkl', 'rb') as f:
    crimeType = pickle.load(f)
with open('./models/Severity.pkl', 'rb') as f:
    sever = pickle.load(f)
with open('./models/gender.pkl', 'rb') as f:
    genderModel = pickle.load(f)

#Files eekada
with open('./mappings/Premis_mapping.json', 'r') as f:
    premis = json.load(f)
with open('./mappings/Crime_mapping.json', 'r') as f:
    crm_mapping = json.load(f)
with open('./mappings/Area_mapping.json', 'r') as f:
    area_mapping = json.load(f)
with open('./mappings/Weapon_mapping.json', 'r') as f:
    weapon_mapping = json.load(f)
with open('./mappings/Desc_mapping.json', 'r') as f:
    desc_mapping = json.load(f)
with open('./mappings/Desc_mapping(W-WO).json', 'r') as f:
    desc = json.load(f)

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    days = ['Monday', 'Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday',]

    month = ['January','Febuary','March','April','May','June','July','August','September','October','November','December']

    letter_to_number = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
        'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19,
        'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26
    }
    # areas = {1: 'Central',2: 'Rampart',3: 'Southwest',4: 'Hollenbeck',5: 'Harbor',6: 'Hollywood',7: 'Wilshire',8: 'West LA',9: 'Van Nuys',10: 'West Valley',11: 'Northeast',12: '77th Street',13: 'Newton',14: 'Pacific',15: 'N Hollywood',16: 'Foothill',17: 'Devonshire',18: 'Southeast',19: 'Mission',20: 'Olympic',21: 'Topanga'}

    return render_template("./index.html",days = days,month = month,areas = area_mapping,premis = premis,weapon = weapon_mapping,desc1 = desc_mapping,desc2 = desc,crime = crm_mapping)

@app.route('/type', methods=['GET', 'POST'])
def predict():
    X = {
    'Hours':request.form['hours'], 'Month': request.form['month'],'Day of Week': request.form['days'], 'AREA': request.form['area'], 'Premis Cd': request.form['premis'], 'Part 1-2': request.form['sev'],'Vict_Sex_Cat':request.form['sex']
    }
    X = pd.DataFrame([X])
    # X_train_scaled = scaler.fit_transform(X)
    results = crimeType.predict(X).tolist()
    return render_template('result.html',x = crm_mapping[str(results[0])],page = 1)
@app.route('/severity', methods=['GET', 'POST'])
def severity():
    X = {
    'Weapon Used Cd': request.form['weapon'],
    'Crm Cd': request.form['crime'],
    'Vict Age': request.form['age'],
    'Vict_Sex_Cat': request.form['sex'],
    'Vict_Desc_Cat': request.form['desc'],
    'AREA': request.form['area']
    }
    X = pd.DataFrame([X])
    results = sever.predict(X).tolist()
    return render_template('result.html',x = results[0],page = 2)
@app.route('/areaPredict', methods=['GET', 'POST'])
def areaPredict():
    X = {
        'LAT': request.form['lat'],
        'LON': request.form['lon']
    }
    X = pd.DataFrame([X])
    results = areaModel.predict(X).tolist()
    return render_template('result.html',x = area_mapping[str(results[0])],page = 3)
@app.route('/genderPredict', methods=['GET', 'POST'])
def genderPredict():
    X = {'Crm Cd': request.form['crime'], 'AREA': request.form['area'], 'Time Period': request.form['time'], 'Vict_Desc_Cat': request.form['desc'], 'Vict Age': request.form['age']}
    X = pd.DataFrame([X])
    results = genderModel.predict(X).tolist()
    gen = ''
    if results[0] == 0:
        gen = 'Female'
    elif results[0] == 1:
        gen = 'Male'
    else:
        gen = 'Not Sure'
    return render_template('result.html',x = gen,page = 4)
if __name__=='__main__':
    app.run()