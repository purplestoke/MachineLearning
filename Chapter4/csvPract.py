import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np


csv_data = """
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
"""

df = pd.read_csv(StringIO(csv_data))
#print(df)
# NUMPY ARRAY
#print(df.values)

# DROP ALL ROWS MISSING A VALUE
#print(df.dropna(axis=0))

# DROP ALL COLUMNS MISSING A VALUE
#print(df.dropna(axis=1))

# DROP ROW WHERE ALL COLUMNS ARE A NaN
#print(df.dropna(how='all'))

# DROP ROWS BASED ON A THRESHOLD
#print(df.dropna(thresh=4))

# DROP ROWS IF NaN IN COLUMN
#print(df.dropna(subset=['C']))

# -----------------------------------------------
# USING THE SIMPLE IMPUTER

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
print(df.fillna(df.mean()))