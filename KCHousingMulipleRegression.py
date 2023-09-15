import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

df = pd.read_csv("../../data/kc_house_data_Poonam.csv")
# id,date,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15
x = df[['sqft_living','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_renovated','lat','sqft_living15']]
y = df[['price']]

x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8)
regressor = LinearRegression()
regressor.fit(x_train,y_train )
test_predicted = regressor.predict(x_test)
print("Regressor Train Score", regressor.score(x_train, y_train))
print("Regressor Test Score", regressor.score(x_test, y_test))
print("Regressor Co-efficient",regressor.coef_) # How much the data is contrubiting to the results
print("Difference between Test and Predicted",mean_absolute_error(y_test, test_predicted))
print("#--------Ridge Regulirazation------------")
ridge = Ridge(alpha=1)
ridge.fit(x_train,y_train )
print("Ridge Train Score", ridge.score(x_train, y_train))
print("Ridge Test  Score", ridge.score(x_test, y_test))
print("Ridge Co-efficient",ridge.coef_) # How much the data is contrubiting to the results
#---------Lesso Regression----------------