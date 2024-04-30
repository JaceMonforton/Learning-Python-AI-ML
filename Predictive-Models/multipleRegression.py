#example: price = alpha + B1(Mileage) + B2(Age) + B3(BodyStyle)
# - Multiple Variables can effect price 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

MileageAndPrice = df[['Mileage' , 'Price']]

bins = np.arange(0, 50000, 10000) #cuts into chunks between (0, 50,000) in 10,000 ranges 
groups = MileageAndPrice.groupby(pd.cut(MileageAndPrice['Mileage'], bins)).mean()
print(groups.head().round(2))
groups['Price'].plot.line()
# plt.show()


scale = StandardScaler()

x = df[['Mileage', 'Cylinder' , 'Doors']]
y = df[['Price']]

x[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(x[['Mileage', 'Cylinder', 'Doors']].values)

#Constant for Y intercept.
x = sm.add_constant(x)
print(x)

est = sm.OLS(y, x).fit() #OLS Regression 
print(est.summary()) #Look @ Coeficients for B1, B1, B3, etc. # -----------------------
                                                                # const       2.134e+04    
                                                                # Mileage    -1272.3412   
                                                                # Cylinder    5587.4472   
                                                                # Doors      -1404.5513  
                                                                

# print(y.groupby(df.Doors).mean())

#Activity

scaled = scale.transform([[1000, 8, 2]]) #10000 miles, 8 cylinders, 2 doors

scaled = np.insert(scaled[0],0,1) # constant column
print("Constant,     Mileage,     Cyl,      Doors")
print(scaled)

predicted = est.predict(scaled)
print("Predicted Price : $" , predicted.round(2))