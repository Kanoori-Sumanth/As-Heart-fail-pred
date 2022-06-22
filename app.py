from flask import Flask, render_template, request
import pickle
################################################################
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_unseen(x):
    prob = model.predict_proba([x])[0][1]
    prob = round(prob, 2)*100
    label = model.predict([x])[0]
    return label, prob
####################### Flask settings #######################
app = Flask(__name__)

####################### Home Page ############################
@app.route("/", methods=['GET'])
def home():
   return render_template("index.html")

########################## Results Page #######################
@app.route("/result", methods = ["GET","POST"])
def predict():
    if request.method == 'POST':
        feature_dict = request.form
        print(feature_dict)
        age = float(feature_dict['age'])
        anaemia = float(feature_dict['anaemia'])
        creatinine_phosphokinase = float(feature_dict['creatinine_phosphokinase'])
        diabetes = float(feature_dict['diabetes'])
        ejection_fraction = float(feature_dict['ejection_fraction'])
        high_blood_pressure = float(feature_dict['high_blood_pressure'])
        platelets = float(feature_dict['platelets'])
        serum_creatinine = float(feature_dict['serum_creatinine'])
        serum_sodium = float(feature_dict['serum_sodium'])
        sex = float(feature_dict['sex'])
        smoking = float(feature_dict['smoking'])

        features = [age,
                    anaemia,
                    creatinine_phosphokinase,
                    diabetes,
                    ejection_fraction,
                    high_blood_pressure,
                    platelets,
                    serum_creatinine,
                    serum_sodium,
                    sex,
                    smoking]

        label, proba= predict_unseen(features)

        return render_template("index.html", proba=proba, label=label)
####################################################

if __name__ == '__main__':

	app.run(debug=True)