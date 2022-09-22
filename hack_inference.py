import pandas as pd
import pickle
from flask import Flask
from flask import request, jsonify

LOCAL_RUN = True
AWS_PORT = 8080

app = Flask(__name__)
rf = pickle.load(open("housing.pkl", "rb"))
oe_dict = pickle.load(open("oe_dict.pkl", "rb"))
encoder = pickle.load(open("ohe_encoder.pkl", "rb"))
nominal_cols = pickle.load(open("nominal_cols.pkl", "rb"))


def process_data(input_df):
    # # transforming ordinal features
    # for col in input_df.columns:
    #     if col in oe_dict:
    #         input_df[col] = oe_dict[col].transform(pd.DataFrame(input_df[col]))

    # transforming Nominal features
    # create df of ohe data
    encoded = pd.DataFrame(encoder.transform(input_df[nominal_cols]), index=input_df.index)
    # drop original nominal columns
    input_df_selected = input_df.drop(columns=nominal_cols)
    # join the encoded columns
    input_df_selected = pd.concat([input_df_selected, encoded], axis=1)

    return input_df_selected


def predict_sample(input_sample):
    y_pred = rf.predict(input_sample)
    return y_pred


@app.route("/predict")
def predict():
    YearRemodAdd = request.args.get("YearRemodAdd")
    ndFlrSF = request.args.get("2ndFlrSF")
    TotalBsmtSF = request.args.get("TotalBsmtSF")
    GrLivArea = request.args.get("GrLivArea")
    stFlrSF = request.args.get("1stFlrSF")
    GarageArea = request.args.get("GarageArea")
    OverallQual = request.args.get("OverallQual")
    # BsmtUnfSF = request.args.get("BsmtUnfSF")
    LotArea = request.args.get("LotArea")
    GarageCars = request.args.get("GarageCars")
    YearBuilt = request.args.get("YearBuilt")
    TotRmsAbvGrd = request.args.get("TotRmsAbvGrd")
    BldgType = request.args.get("BldgType")
    Neighborhood = request.args.get("Neighborhood")
    input = pd.DataFrame(data=[[LotArea, TotalBsmtSF, OverallQual, GarageCars, GarageArea, GrLivArea, YearRemodAdd,
                                ndFlrSF, stFlrSF, YearBuilt, TotRmsAbvGrd, BldgType, Neighborhood]],
                         columns=['LotArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars', 'GarageArea', 'GrLivArea', 'YearRemodAdd',
                                '2ndFlrSF', '1stFlrSF', 'YearBuilt', 'TotRmsAbvGrd', 'BldgType', 'Neighborhood'])
    processed_data = process_data(input)
    pred = predict_sample(processed_data)
    return str(pred[0])


if __name__ == '__main__':
    if LOCAL_RUN:
     app.run(debug=True)
    else:
     app.run(host="0.0.0.0", port=AWS_PORT, debug=True)
