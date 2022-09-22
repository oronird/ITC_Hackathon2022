import pandas as pd
import pickle
from flask import Flask
from flask import request, jsonify

LOCAL_RUN = False
AWS_PORT = 8080

app = Flask(__name__)
rf = pickle.load(open("housing.pkl", "rb"))


def predict_sample(input_sample):
    y_pred = rf.predict(input_sample)
    return y_pred


@app.route("/predict")
def predict():
    GarageYrBlt = request.args.get("GarageYrBlt")
    ndFlrSF = request.args.get("2ndFlrSF")
    TotalBsmtSF = request.args.get("TotalBsmtSF")
    GrLivArea = request.args.get("GrLivArea")
    stFlrSF = request.args.get("1stFlrSF")
    GarageArea = request.args.get("GarageArea")
    OverallQual = request.args.get("OverallQual")
    BsmtUnfSF = request.args.get("BsmtUnfSF")
    LotArea = request.args.get("LotArea")
    GarageCars = request.args.get("GarageCars")
    input = pd.DataFrame(data=[[GarageYrBlt, ndFlrSF, TotalBsmtSF, GrLivArea, stFlrSF, GarageArea, OverallQual,
                                BsmtUnfSF, LotArea, GarageCars]], columns=['GarageYrBlt', 'ndFlrSF', 'TotalBsmtSF',
                                                                           'GrLivArea', 'stFlrSF', 'GarageArea',
                                                                           'OverallQual', 'BsmtUnfSF', 'LotArea',
                                                                           'GarageCars'])
    pred = predict_sample(input)
    return str(pred[0])


if __name__ == '__main__':
    if LOCAL_RUN:
     app.run(debug=True)
    else:
     app.run(host="0.0.0.0", port=AWS_PORT, debug=True)
