import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv')

gear_counts = df['# Gears'].value_counts()

gear_counts.plot(kind='bar')
plt.show()

print(df.head())

sns.distplot(df["CombMPG"])
plt.show()

df2 = df[['Cylinders', 'CityMPG' , 'HwyMPG', 'CombMPG']]
print(df2.head())
sns.pairplot(df2 , height=2.5)
plt.show()


ax = sns.boxenplot(x='Mfr Name', y='CombMPG', data = df)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.show()

yx = sns.stripplot(x='Mfr Name', y='CombMPG', data = df) #or swamplot
yx.set_xticklabels(ax.get_xticklabels(), rotation = 45)

plt.show()

# Exercise

val = sns.scatterplot(x='# Gears' , y='CombMPG' , data=df)
plt.show()
val = sns.lmplot(x='# Gears' , y='CombMPG' , data=df)
plt.show()
val = sns.jointplot(x='# Gears' , y='CombMPG' , data=df)
plt.show()
val = sns.boxplot(x='# Gears' , y='CombMPG' , data=df)
plt.show()
val = sns.swarmplot(x='# Gears' , y='CombMPG' , data=df)
plt.show()

#Draw conclusion that cars with 6 gears will typically have better MPG 
