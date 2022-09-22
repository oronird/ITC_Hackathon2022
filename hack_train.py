import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# read data and remove all samples from 2010
df = pd.read_csv('train.csv')
df = df.loc[df['YrSold'] != 2010, :]

# columns with missing values
missing_cols = df.isna().sum(axis=0)[df.isna().sum(axis=0) > 0].index
missing_cols_cat = list(missing_cols)

# removing numerical columns with missing values
missing_cols_cat.remove('GarageYrBlt')
missing_cols_cat.remove('MasVnrArea')
missing_cols_cat.remove('LotFrontage')

# imputing mean values for numerical features
missing_cols_num = ['GarageYrBlt', 'MasVnrArea', 'LotFrontage']
df.loc[:,missing_cols_cat] = df.loc[:,missing_cols_cat].fillna(value='unknown')
for col in missing_cols_num:
    df[col] = df[col].fillna(df[col].mean())

numeric_cols = list(df.select_dtypes(exclude='object').columns)
catagorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

ordinal_cols_dict = {'LotShape' : ["Reg", "IR1", "IR2", "IR3"],
'ExterQual' : ["Ex", "Gd","TA","Fa", "Po"],
'ExterCond' : ["Ex", "Gd","TA","Fa", "Po"],
'BsmtQual' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'BsmtCond' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'BsmtExposure' : ["Gd", "Av", "Mn", "No", "NA"],
'BsmtFinType1' : ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
'BsmtFinType2' : ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
'HeatingQC' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'KitchenQual' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'Functional' : ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
'FireplaceQu' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'GarageFinish' : ["Fin", "RFn", "Unf", "NA"],
'GarageQual' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'GarageCond' : ["Ex", "Gd","TA","Fa", "Po", "NA"],
'PoolQC' : ["Ex", "Gd","TA","Fa", "Po", "NA"]}

for col in missing_cols_cat:
    if col in ordinal_cols_dict.keys():
        ordinal_cols_dict[col].append("unknown")

ordinal_cols = ['LotShape', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'HeatingQC' , 'KitchenQual' ,'Functional' ,'FireplaceQu', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PoolQC']

oe_dict = {}
for key in ordinal_cols_dict.keys():
    oe = OrdinalEncoder(categories=[ordinal_cols_dict[key]])
    oe.fit(df[[key]])
    oe_dict[key] = oe
    df[key] = oe.transform(pd.DataFrame(df[key]))

pickle.dump(oe_dict, open("oe_dict.pkl", "wb"))

nominal_cols = list(set(catagorical_cols) - set(ordinal_cols))

selected_cols = ['LotArea',
  'TotalBsmtSF',
  'OverallQual',
  'GarageCars',
  'GarageArea',
  'GrLivArea',
  'YearRemodAdd',
  '2ndFlrSF',
  '1stFlrSF',
  'YearBuilt',
  'TotRmsAbvGrd',
  'BldgType',
  'Neighborhood']

nominal_cols = list(set(nominal_cols) & set(selected_cols))

df_selected = df[selected_cols]

# define one hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# fit & transform data
encoder.fit(df_selected[nominal_cols])
# create df of ohe data
encoded = pd.DataFrame(encoder.transform(df_selected[nominal_cols]), index=df_selected.index)
# drop original nominal columns
df_selected = df_selected.drop(columns=nominal_cols)
# join the encoded columns
df_selected = pd.concat([df_selected, encoded], axis=1)

pickle.dump(encoder, open("ohe_encoder.pkl", "wb"))
pickle.dump(nominal_cols, open("nominal_cols.pkl", "wb"))

target = 'SalePrice'
# split to X and y
X = df_selected
y = df[target]


rf = RandomForestRegressor()
rf.fit(X,y)

pickle.dump(rf, open("housing.pkl", "wb"))