from numpy import random

random.seed(0)

totals = {20:0, 30:0, 40: 0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}

totalPurchases = 0

for _ in range(100000):
    ageDecade = random.choice([20,30,40,50,60,70])
    purchaseProb = 0.4
    totals[ageDecade] += 1
    if (random.random() < purchaseProb):
        totalPurchases += 1
        purchases[ageDecade] += 1

print(totals) # age has no effect
print(purchases) # increases based on age, relationship between age and likelyhood of purchasing something.


PEF = float(purchases[30]) / float(totals[30]) #prob (Prob of purchasing | youre 30)
print("P(Purchase | 30): " ,str(PEF))

PF = float(totals[30]) / 100000.0
print("P(30): " ,str(PF))

PE = float(totalPurchases) / 100000.0
print("P(Purchase): " ,str(PE))
