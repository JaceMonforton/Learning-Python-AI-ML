import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
from io import StringIO
import pydot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import graphviz
input_file = "C:\MLCourse\PastHires.csv"

df = pd.read_csv(input_file, header=0)

d = {"Y": 1, "N": 0}
df['Hired'] = df['Hired'].map(d)  # changes Y to 1, N to 0
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)

d = {"BS": 0, "MS": 1, "PhD": 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())

features = list(df.columns[:6])
print(features)

y = df['Hired']
X = df[features]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
(graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())

# Save decision tree as PNG
graph.write_png("decision_tree1.png")

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)

# Predict employment for a sample input
sample_input = np.array([[10, 0, 4, 0, 0, 0]])  # Example input data\


prediction = clf.predict(sample_input)
print("Predicted employment:", prediction)
