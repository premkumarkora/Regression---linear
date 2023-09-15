from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("../../data/Salary_Data.csv")
#Input will be number of hours Output will be the corresponding hours
# iloc[startrow:end row, start column : end column]
#print(df)
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]
print(x,y)
x_train,x_test,y_train,y_test =train_test_split(x.values,y.values,train_size=0.8)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
# y_predicted is to check the error margin
x_pred = regressor.predict(x_train)
y_pred = regressor.predict(x_test)

years = float(input(" Please let us know your experience to predict your salary :"))
y_predicted1 = regressor.predict([[years]])

plt.scatter(years,y_predicted1, color="red")
plt.scatter(x_train,y_train, color="green")
plt.plot(x_test, y_pred)
plt.show()

print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient",regressor.coef_) # How much the data is contrubiting to the results
print("Difference between Test and Predicted",mean_absolute_error(y_test, y_pred))
