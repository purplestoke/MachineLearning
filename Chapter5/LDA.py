import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

# COMPUTING THE MEAN VECTORS

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))

# COMPUTING THE WITHIN CLASS SCATTER MATRIX
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d,d))
for row in X_train_std[y_train == label]:
    row, mv = row.reshape(d, 1), mv.reshape(d,1)
    class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
print('Class label distribution: %s' % np.bincount(y_train)[1:])

# COMPUTING THE SCALED WITHIN-CLASS SCATTER MATRIX
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter= np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print('Scaled within-class scatter matrix %sx%s' % (S_W.shape[0], S_W.shape[1]))

# NOW WE COMPUTE THE BETWEEN-CLASS SCATTER MATRIX 
mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between class scatter matrix %sx%s' % (S_B.shape[0], S_B.shape[1]))

# FIND EIGENVALS AND EIGENVECS 
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# SORT IN DESC ORDER
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigen Values in desc order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# DISCRIMINABILITY PLOT

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha=0.5, align='center', label='Individual discriminability')
plt.step(range(1,14), cum_discr, where='mid', label='cumulative discriminability')
plt.ylabel('Discriminability ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# STACK THE 2 MOST DISCRIMINATIVE EIGENVECTOR COLUMNS TO CREATE THE TRANSFORMATION MATRIX W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

# PROJECTING FEATURES ONTO NEW SUBSPACE 
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train ==l, 0],
                X_train_lda[y_train==l, 1] * (-1),
                c=c, label=l, marker=m)
    
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()