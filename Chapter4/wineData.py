import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# WINE DATASET
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None
)
#   COLUMN LABELS
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavonoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', "Proline"]


#print('Class labels', np.unique(df_wine['Class label']))
#print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# APPLY NORMALIZATION TO OUR DATA ------------------------------------
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
#print(X_train_norm)
#print(X_test_norm)

# APPLY STANDARDIZATION TO OUR DATA -------------------------------------
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
#print(X_train_std)
#print(X_test_std)

# L1 REGULARIZATION
from sklearn.linear_model import LogisticRegression

#lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')

#lr.fit(X_train_std, y_train)
#print("Training Accuracy:", lr.score(X_train_std, y_train))
#print("Test accuracy:", lr.score(X_test_std, y_test))
#print(lr.intercept_)
#print(lr.coef_)

# PLOT THE REGULARIZATION PATH
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)
colors = [
    'blue', 'green', 'red', 'cyan',
    'magenta', 'yellow', 'black',
    'pink', 'lightgreen', 'lightblue',
    'gray', 'indigo', 'orange'
]
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xscale('log') 
plt.xlim([1e-5, 1e5]) 
plt.ylabel('weight coef')
plt.xlabel('C')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
#plt.show()