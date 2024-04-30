import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, binom

#Uniform Dist
values = np.random.uniform(-10.0, 10.0, 10000)

plt.hist(values, 50)
# plt.show()

# # Normal Dist
x = np.arange(-3, 3, 0.001) # x values with domain(-3, 3)
plt.plot(x, norm.pdf(x))
plt.show()
# #-----------------------------------------#
mu = 5.0
sigma = 2.0
values = np.random.normal(mu, sigma, 10000)
plt.hist(values, 50)
plt.show()

# Exponential 
x = np.arange(0, 10, 0.001) #between 0-10 , step size of 0.001
plt.plot(x, expon.pdf(x))
plt.show()

#Binomial
n, p = 10, 0.5

x = np.arange(0, 10 , 0.001)

plt.plot(x, binom.pmf(x, n, p))
plt.show()

