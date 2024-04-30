import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


pageSeeds = np.random.normal(3.0, 1.0,100)

purchaseAmt = np.random.normal(50.0, 30.0, 100) / pageSeeds

plt.scatter(pageSeeds, purchaseAmt)


trainX = pageSeeds[:80]
testX = pageSeeds[80:]

trainY = purchaseAmt[:80]
testY = purchaseAmt[80:]

# plt.scatter(trainX, trainY)
# plt.show()
# plt.scatter(testX, testY)
# plt.show()


#Train Data
x = np.array(trainX)
y = np.array(trainY)
p4 = np.poly1d(np.polyfit(x, y, 8))

xp = np.linspace(0, 7, 100)
axes = plt.axes()

axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(x,y)
plt.plot(xp, p4(xp), c='r')
plt.show()

r_squared = r2_score(y, p4(x))
if (r_squared <= 1 and r_squared >= -1):
    print("Trained R-Val: " ,r_squared)
else:
    print("Error")

#Test Data
xTest = np.array(testX)
yTest = np.array(testY)
p4 = np.poly1d(np.polyfit(x, y, 5))

xpTest = np.linspace(0, 7, 100)
axes = plt.axes()

axes.set_xlim([0,7])
axes.set_ylim([0,200])
plt.scatter(xTest,yTest)
plt.plot(xpTest, p4(xpTest), c='r')
plt.show()

r_squaredTest = r2_score(testY, p4(testX))
if (r_squaredTest <= 1 and r_squaredTest >= -1):
    print("Test R-Val: " ,r_squaredTest)
else:
    print("Error")