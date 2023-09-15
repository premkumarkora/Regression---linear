#Unsupervised Outlier Detection. Estimate the support of a high-dimensional distribution.
from sklearn.svm import OneClassSVM
X = [[0], [0.44], [0.45], [0.46], [1]]
clf = OneClassSVM(gamma='auto').fit(X)
clf.predict(X)

print(clf.score_samples(X))
