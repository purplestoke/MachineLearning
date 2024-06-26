import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# WINE DATASET
df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None
)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# STANDARDIZE THE FEATURES
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# DOING SOME LINEAR ALGEBRA EIGENVALUES, VECTORS AND PAIRS

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
#print('\nEigenValues \n%s' % eigen_vals)

# PLOT THE CUMSUM OF EXPLAINED VARIANCES
import matplotlib.pyplot as plt

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_expl = np.cumsum(var_exp)
"""
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_expl, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
#plt.show()"""

# MAKE A LST OF EIGEN-VALUE, EIGEN-VECTOR TUPLES
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# SORT TUPLES FROM HIGH TO LOW
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0] [1] [:, np.newaxis], 
               eigen_pairs[1] [1] [:, np.newaxis]))

#print('Matrix W:\n', w)


# PROJECTING THE WINE DF ONTO OUR MATRIX W
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0],
                X_train_pca[y_train==l, 1],
                c=c, label=l, marker=m)
    
plt.xlabel("PC 1")
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()