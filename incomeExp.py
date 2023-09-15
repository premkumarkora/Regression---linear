from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df= pd.DataFrame()
df["Income"] = [100,200,100,300,150,230,220,500,250,310,400,350,375,595,120,150]
df["Expense"] = [70,120,80,150,120,200,180,350,175,210,250,280,290,400,79,140]
#print(df)

x = df.iloc[:,:1]
y = df.iloc[:,-1:]
#print(x,y)


x_train, x_test,y_train,y_test = train_test_split(x,y,train_size=0.70)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
y_train_predicted = regressor.predict(x_train)
#x_pred = regressor.predict(x_train)
y_test_predicted= regressor.predict(x_test)

income = float(input(" Please let us know your Income to predict your expense :"))
y_predicted_expense = regressor.predict([[income]])

plt.scatter(x_train,y_train, color="red")
plt.scatter(income,y_predicted_expense, color="blue")
plt.plot(x_train, y_train_predicted)
plt.show()

print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient",regressor.coef_) # How much the data is contrubiting to the results
#print("Difference between Test and Predicted",mean_absolute_error(y_test, y_predicted))


print("Using Numpy method for Train:", np.sqrt(mean_squared_error(y_train,y_train_predicted)))
print("Using r2_score method  for Train:",r2_score(y_train, y_train_predicted))

print("Using Numpy method for Test:",np.sqrt(mean_squared_error(y_test,y_test_predicted)))
print("Using r2_score method  for Test:",r2_score(y_test, y_test_predicted))


x1=df['Income'].values
y1=df['Expense'].values
slope,intercept,a,b,c = stats.linregress(x1,y1)
print("Slope : ", slope,"Intercept :",intercept)


'''
The input variables are assumed to have a Gaussian distribution(the normal range is
 defined as the range of values lying between the limits specified by 
 two standard deviations below the mean and two standard deviations )
 
 y = a1x1 + a2x2 + a3x3 + ..... + anxn + b
 
y is the target variable.
x1, x2, x3,...xn are the features.
a1, a2, a3,..., an are the coefficients.
b is the parameter of the model.
'''