import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 1000)

purchaseAmt = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

plt.scatter(pageSpeeds, purchaseAmt)

x = np.array(pageSpeeds)
y = np.array(purchaseAmt)
p4 = np.poly1d(np.polyfit(x,y, 4)) #4th deg polynomial fit.
xp = np.linspace(0,7,100)

plt.scatter(x,y)
plt.plot(xp, p4(xp), c='r')
plt.show()


r2 = r2_score(y, p4(x))
print("R squared value: ",r2)

