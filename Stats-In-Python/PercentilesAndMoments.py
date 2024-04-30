import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

#Percentiles
vals = np.random.normal(0, 0.5, 10000)
plt.hist(vals, 50)
# plt.show()

print("50th Percentile: " , np.percentile(vals, 50))
print("Median: ", np.median(vals))
print("20th Percentile: " , np.percentile(vals, 20))
print("99th Percentile: " , np.percentile(vals, 99))


#Moments
np.var(vals)
np.mean(vals)

skew = sp.skew(vals)
kurtosis = sp.kurtosis(vals)
print("skew = " , skew)
print("kurtosis = " , kurtosis)



