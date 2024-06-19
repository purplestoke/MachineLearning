import pandas as pd
import numpy as np

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3,
                "L": 2,
                'M': 1}
inverse_mapping = {v: k for k,v in size_mapping.items()}
df['size'] = df['size'].map(size_mapping)

#class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
#inv_class_mapping = {v: k for k, v in class_mapping.items()}
#df['classlabel'] = df['classlabel'].map(class_mapping)

# -------------------------------------------------------------------
# SCIKIT-LEARNS WAY OF ENCODING CLASS LABELS
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
#y = class_le.fit_transform(df['classlabel'].values)
#print(y)

#INVERSE
#print(class_le.inverse_transform(y))


# CAN USE LABEL ENCODER FOR COLOR COLUMN
#X = df[['color', 'size', 'price']].values
#color_le = LabelEncoder()
#X[:, 0] = color_le.fit_transform(X[:, 0])
#print(X)

# --------------------------------------------------------------------
# ONE HOT ENCODING USING SCIKIT-LEARN

from sklearn.preprocessing import OneHotEncoder

#X = df[['color', 'size', 'price']].values
#color_ohe = OneHotEncoder()
#print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())


# -----------------------------------------------------------------------
# SELECTIVELY TRANSFORM COLUMNS
from sklearn.compose import ColumnTransformer 

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
#print(c_transf.fit_transform(X).astype(float))

# --------------------------------------------------------------------
# PANDAS METHOD OF ONE HOT ENCODING
#print(pd.get_dummies(df[['price', 'color', 'size']]))

# DROP FIRST COLUMN OF OHE
#print(pd.get_dummies(df[['price', 'color', 'size']],
#               drop_first=True))

# DROP REDUNDANT COLUMNS
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1, 2])
])
print(c_transf.fit_transform(X).astype(float))