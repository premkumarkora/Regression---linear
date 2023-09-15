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
df["x"] = [1,2,3,4,5,6,7,8]
df["y"] = [3.6,3.8,2,3.8,4,4,4.1,3]

#df["x"] = [1,2,3,4,5,6,7,8]
#df["y"] = [3,4,2,4,5,7,5,3]
#print(df)
x = df.iloc[:,:1]
y = df.iloc[:,-1:]

x_train, x_test,y_train,y_test = train_test_split(x,y,train_size=0.80, random_state=1)
regressor = LinearRegression()
regressor.fit(x_train,y_train )

y_pred = regressor.predict(x_train)


y_pred_forR2 = regressor.predict(x) # Predicting values for all x to find R2
#print("All predicted Values of y : ", y_pred_forR2)
print("R^2(Square) values for all values:", metrics.r2_score(y, y_pred_forR2))

# model evaluation for testing set

maeTrain = metrics.mean_absolute_error(x_train, y_train)
mseTrain = metrics.mean_squared_error(x_train, x_train)
maeTest = metrics.mean_absolute_error(x_test, y_test)
mseTest = metrics.mean_squared_error(x_test, y_test)
r2Train = metrics.r2_score(x_train, y_train)
r2Test = metrics.r2_score(x_test, y_test)

print("The model performance for testing set")
print("--------------------------------------")
print('Mean absolute error Train is {}'.format(maeTrain))
print('Mean squared error Train is {}'.format(mseTrain))
print('Mean absolute error Test is {}'.format(maeTest))
print('Mean squared error Test is {}'.format(mseTest))
print('R2 score for Train {}'.format(r2Train))
print('R2 score for Test {}'.format(r2Test))


print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient is M i.e. slope = ",regressor.coef_)
print("Regressor Intercept is C i.e. Y-Intercept = ",regressor.intercept_)


x1=df['x'].values
y1=df['y'].values
slope,intercept,a,b,c = stats.linregress(x1,y1)
print("Slope M : ", slope,"Intercept @ y axis:",intercept)

plt.scatter(x_train,y_train, color="red")
plt.plot(x_train, y_pred)
plt.show()

'''
Mean absolute error
Small MAE suggests the model is great at prediction. 
Large MAE suggests that your model may have trouble at generalizing well. 
An MAE of 0 means that our model outputs perfect predictions, but this is unlikely to happen in real scenarios.


Mean squared error - 
 MSE will almost always be bigger than MAE because in MAE residuals contribute linearly to the total error, 
 while in MSE the error grows quadratically with each residual. 
 This is why MSE is used to determine the extent to which the model fits the data because 
 it strongly penalizes the heavy outliers.

R^2(Square) can take values from 0 to 1. 
A value of 1 indicates that the regression predictions perfectly fit the data.

 Tips For Using Regression Metrics
We always need to make sure that the evaluation metric we choose for a regression problem 
does penalize errors in a way that reflects the consequences of those errors for the business, 
organizational, or user needs of our application.

If there are outliers in the data, they can have an unwanted influence on the overall R^2 or MSE scores. 
MAE is robust to the presence of outliers because it uses the absolute value. 
Hence, we can use the MAE score if ignoring outliers is important to us.


MAE is the best metrics when we want to make a distinction between different models 
because it doesn’t reflect large residuals.

If we want to ensure that our model takes the outliers into account more,
we should use the MSE metrics.
'''