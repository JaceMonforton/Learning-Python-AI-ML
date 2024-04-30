import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# incomes = np.random.normal(27000,15000,10000)
# incomes = np.append(incomes,[1000000000])  #Add outlier 
# print("Mean: ", np.mean(incomes))
# print("Median: ", np.median(incomes))

# plt.hist(incomes, 50) #histogram of data set, 50 diff buckets
# plt.show() #shows histogram in gui


# ages = np.random.randint(18, high=90, size=500)
# # print(ages)
# print(stats.mode(ages))
print('============================')
#Exercise

incomes = np.random.normal(100, 20, 10000)
plt.hist(incomes, 50)
# plt.show()

#Todo: Find mean median and mode, check result

median = np.median(incomes)
mean = np.mean(incomes)

print("Mean: ", mean)
print("Median: ", median)

mode = stats.mode(incomes)
print("Mode: ", mode)

#------------------------#
StandardDev =incomes.std()
Variance = incomes.var()
print(StandardDev, Variance)



