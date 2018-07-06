#! /usr/bin/python

import matplotlib.pyplot as plt
# %matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
from tensorflow.contrib import learn

import seaborn

# READ INPUT DATA
titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])

titanic_df['pclass'] = titanic_df['pclass'].astype(object)

# SHOW DATA (Top Part)
titanic_df.head()

# SHOW MEAN of SURVIVED
titanic_df['survived'].mean()

# SHOW Grouping With Ticket Class (first, second, third class)
titanic_df.groupby('pclass').mean()

# SHOW TWO GROUPINGS
class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping

class_sex_grouping['survived'].plot.bar()

plt.savefig("1.png")
plt.show()

# SHOW AGE CATEGORIZATION
group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()

plt.savefig("2.png")
plt.show()

# DATA PREPARATION

# Look at the column counts to see missing data
titanic_df.count()

# Drop unrelated or too much missing data columns
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)

# Fill N/A to missing values
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")

# Drop N/A Values
titanic_df = titanic_df.dropna()

# See the column count after cleaning
titanic_df.count()


# PROCESS DATA FOR MACHINE LEARNING ALGORITHM


# Preprocess to put numbers instead of string values

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.pclass = le.fit_transform(processed_df.pclass)
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
    return processed_df

processed_df = preprocess_titanic_df(titanic_df)

# Set X and Y
X = processed_df.drop(['survived'], axis=1)
y = processed_df['survived']

# Set training data (0.2 means %80 of data used for training)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X.values,y.values,test_size=0.2)

seaborn.heatmap(processed_df.corr(), annot=True, cmap="RdYlGn")
plt.show()

###################################################################################################
# DECISION TREE

print("Decision Tree")

clf_dt = tree.DecisionTreeClassifier(max_depth=10)


# Train decision tree
clf_dt.fit(X_train, y_train)

# Score with test data
clf_dt.score(X_test,  y_test)


# Shuffle Validator (to train with different random sample from data set)

shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)

def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X.values, y.values, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))

test_classifier(clf_dt)


# SHOW TREE
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf_dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png("tree.png")



###################################################################################################
# RANDOM FOREST
print("Random Forest")

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)


###################################################################################################
# GRADIENT BOOSTER
print("Gradient Booster")

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)


# ###################################################################################################
# # VOTING CLASSIFIER
# print("Voting Classifier")
# 
# eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
# test_classifier(eclf)


feature_columns = []

for key in X.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


print(feature_columns)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(10).repeat().batch(batch_size)
    return dataset

# TENSORFLOW
tf_clf_dnn = tf.estimator.DNNClassifier(hidden_units=[10, 10], feature_columns=feature_columns, n_classes=2)
tf_clf_dnn.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size=100), steps=400 )
tf_clf_dnn.score(X_test, y_test)

