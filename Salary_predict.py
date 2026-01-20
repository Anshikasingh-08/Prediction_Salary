# importing libraries
import pandas as pd
import numpy as np

# loading dataset
df = pd.read_csv('/content/Position_Salaries.csv')

# loading values into X and y
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# printing data
df.head()

# importing Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# implementing Polynomial Linear Regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# plotting linear regression model 
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'brown')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# plotting polynomial regression model 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'brown')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

# predicting salary for certain level individual 
lin_reg_2.predict(poly_reg.fit_transform(np.array(6.9).reshape(1,-1)))