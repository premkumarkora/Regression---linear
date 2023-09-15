import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

df= pd.DataFrame()
df["x"] = [1,2,3,4,5]
df["y"] = [3,4,2,4,5]
x1=df['x'].values
y1=df['y'].values


x = df.iloc[:,:1]
y = df.iloc[:,-1:]

x_train, x_test,y_train,y_test = train_test_split(x,y,train_size=0.67)
regressor = LinearRegression()
regressor = regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_train)

maeTrain = metrics.mean_absolute_error(x_train, y_train)
mseTrain = metrics.mean_squared_error(x_train, x_train)
maeTest = metrics.mean_absolute_error(x_test, y_test)
mseTest = metrics.mean_squared_error(x_test, y_test)
r2Value = metrics.r2_score(x1, y1)
print("mean_absolute_error Train",maeTrain)
print("mean_squared_error Train",mseTrain)
print("mean_absolute_error Test",maeTest)
print("mean_squared_error Test",mseTest)
print("metrics.r2_score for all x and y",r2Value)
print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient is M i.e. slope = ",regressor.coef_)
print("Regressor Intercept is C i.e. Y-Intercept = ",regressor.intercept_)

plt.scatter(x_train,y_train, color="blue")
plt.plot(x_train,y_pred, color="red")
plt.show()