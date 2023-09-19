#-------------------------------------------------------------------------
# AUTHOR: Abanob Ayoub
# FILENAME: decision_tree
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'No': 1, 'Yes': 2}
tear_mapping = {'Reduced': 1, 'Normal': 2}
class_mapping = {'Yes': 1, 'No': 2}

# Reading the data from the CSV file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)

# Transforming the original categorical training features into numbers and adding to the 4D array X
for row in db:
    x_row = [age_mapping[row[0]], spectacle_mapping[row[1]], astigmatism_mapping[row[2]], tear_mapping[row[3]]]
    X.append(x_row)

# Transforming the original categorical training classes into numbers and adding to the vector Y
for row in db:
    Y.append(class_mapping[row[4]])

# Fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# Plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()