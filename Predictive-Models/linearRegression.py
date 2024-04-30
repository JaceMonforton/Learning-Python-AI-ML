import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
pageSpeeds = np.random.normal(3.0, 1.0, 1000)

purchaseAmt = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

# plt.scatter(pageSpeeds, purchaseAmt)
# plt.show()

slope, intercept, r_val, p_val, std_err = stats.linregress(pageSpeeds, purchaseAmt)

r_sqared = r_val**2 

print("Slope: " ,slope)
print("Intercept: " ,intercept)
print("R-Value: " ,r_sqared)

def predict(x):
    return slope * x + intercept #formula for least squares line

fitline = predict(pageSpeeds)
plt.scatter(pageSpeeds, purchaseAmt)
plt.plot(pageSpeeds, fitline, c='r') #plots the scatter plot with line of best fit with color red
plt.show()

def predictAtValue(x):
    return slope * x + intercept

predictedValue = predictAtValue(2.5)
print("Predicted Value: " ,predictedValue)