from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df= pd.DataFrame()
df["x1"] = [1,2,3,4,5]
df["x2"] = [2,5,2,4,3]
df["y"] = [3,4,2,4,5]
#print(df)
x = df[['x1', 'x2']]
y = df[['y']]
print(x)
print(y)
#slope,intercept,a,b,c = stats.linregress(x,y)
#print("Slope M : ", slope,"Intercept @ y axis:",intercept)


x_train, x_test,y_train,y_test = train_test_split(x,y,train_size=0.60)
regressor = LinearRegression()
regressor.fit(x_train,y_train )

y_train_pred = regressor.predict(x_train)
y_test_pred  = regressor.predict(x_test)
#print(x_train)
#print(x_test)
#print(y_train_pred)
#print(y_test_pred)


# model evaluation for testing set

mae = metrics.mean_absolute_error(y_test, y_test_pred)
mse = metrics.mean_squared_error(y_test, y_test_pred)
r2 = metrics.r2_score(y_test, y_test_pred)

print("The model performance for testing Data set only")
print("--------------------------------------")
print('Mean absolute error is {}'.format(mae))
print('Mean squared error is {}'.format(mse))
print('R2 score is {}'.format(r2))


print("--------------------------------------------------")
print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient is M i.e. slope = ",regressor.coef_)
print("Regressor Intercept is C i.e. Y-Intercept = ",regressor.intercept_)




