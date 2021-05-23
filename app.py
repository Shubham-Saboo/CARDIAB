from flask import Flask, request, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("Diabetes1.pkl", "rb"))
model1 = pickle.load(open("model2.pkl", "rb"))
sc = pickle.load(open("scalar1.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/sugar')
def sugar():
    return render_template("sugar.html")


@app.route('/cardio')
def cardio():
    return render_template("cardio.html")


@app.route('/cardiopredict', methods=['POST', 'GET'])
def results():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']

    row_df = pd.DataFrame(
        [pd.Series([text1, text2, text3, text4, text5, text6])])
    
    new_data_scaled = sc.transform(row_df)
    

    prediction = model1.predict_proba(new_data_scaled)

    

    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    output = float(output)*100
    if output > 50.0:
        return render_template('result3.html', prob='Yes', percent=f'{str(output)} %')
    else:
        return render_template('result4.html', prob='No', percent=f'{str(output)} %')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    

    row_df = pd.DataFrame(
        [pd.Series([text1, text2, text3, text4, text5])])

    prediction = model.predict_proba(row_df)

    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    output = float(output)*100
    if output > 50.0:
        return render_template('result1.html', prob='Yes', percent=f'{str(output)} %')
    else:
        return render_template('result2.html', prob='No', percent=f'{str(output)} %')


if __name__ == '__main__':
    app.run(debug=True)
