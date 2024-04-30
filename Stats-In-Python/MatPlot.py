import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random as randn
x = np.arange(-3, 3, 0.001)

plt.plot(x, norm.pdf(x), 'b-') #blue solid
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:') #red dotted
#save to file
# plt.savefig('C:\\Users\\jace1\\OneDrive\\Desktop\\Python-Projects\\Stats-In-Python\\MyPlot.png')

#axis & legends

plt.ylabel("Probability")
plt.xlabel("Number Of Students")
plt.legend(['Graph 1','Graph 2'] , loc=6)
plt.show()


#Scatter plot

x = [1,3,6,5,4,8,6,3,5,2,4,7,7,2,1,234,4,4]
y = [1,3,6,5,4,8,6,3,5,2,4,7,7,2,1,234,4,4]
plt.scatter(x,y)
plt.show()