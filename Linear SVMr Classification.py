#Similar to SVC with parameter kernel=’linear’, This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme.from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
X, y = make_classification(n_features=4, random_state=0)
clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
clf.fit(X, y)
print(clf.named_steps['linearsvc'].coef_)
print(clf.named_steps['linearsvc'].intercept_)

print(clf.predict([[0, 0, 0, 0]]))