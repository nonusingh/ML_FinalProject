#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person']# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "length of the dataset:"
print len(data_dict)

#  What is the Number of features in the dict?
unique_features = set(
    feature
    for row_dict in data_dict.values()
    for feature in row_dict.keys()
)
print "number of unique features:"
print(len(unique_features))
print(unique_features)

# How many POIs in the dataset? How many are not POIs
count = 0
for user in data_dict:
    if data_dict[user]['poi'] == True:
        count+=1
print "number of employees that are 'persons of interest':"
print (count)

print "number of employees that are not POI:"
print len(data_dict)-(count)

### Task 2: Remove outliers
import matplotlib.pyplot
data = featureFormat(data_dict, features_list)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# How to handle outlier with 1e7 bonus?
data_dict.pop('TOTAL', 0)

# How many features have most missing values?
missing_features = {}
for feature in unique_features:
    missing_count = 0
    for k in data_dict.iterkeys():
        if data_dict[k][feature] == "NaN":
            missing_count += 1
    missing_features[feature] = missing_count
print missing_features

### Task 3: Create new feature(s)
# New Feature: from_poi
for employee in data_dict:
    if (data_dict[employee]['to_messages'] not in ['NaN', 0]) and (data_dict[employee]['from_this_person_to_poi'] not in ['NaN', 0]):
        data_dict[employee]['from_poi'] = float(data_dict[employee]['to_messages'])/float(data_dict[employee]['from_this_person_to_poi'])
    else:
        data_dict[employee]['from_poi'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict
features_list = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person','from_poi']

### Is the new feature important?
features_list1 = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person']
features_list2 = ['poi','salary','to_messages', 'deferral_payments',
                'total_payments','exercised_stock_options','bonus','restricted_stock',
                'shared_receipt_with_poi','total_stock_value','expenses','from_messages',
                'other','from_this_person_to_poi','deferred_income','long_term_incentive',
                'from_poi_to_this_person','from_poi']

from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()

### Extract features and labels from dataset for local testing
data1 = featureFormat(my_dataset, features_list1, sort_keys = True)
labels1, features1 = targetFeatureSplit(data1)
data2 = featureFormat(my_dataset, features_list2, sort_keys = True)
labels2, features2 = targetFeatureSplit(data2)

gnb_clf.fit(features1, labels1)
print "Accuracy without new feature:", gnb_clf.score(features1, labels1)
gnb_clf.fit(features2, labels2)
print "Accuracy with new feature:", gnb_clf.score(features2, labels2)

### Top important features
from sklearn.feature_selection import SelectKBest, f_classif
skb = SelectKBest()
selected_features = skb.fit_transform(features2,labels2)
features_selected=[features_list2[i+1] for i in skb.get_support(indices=True)]
print 'Features selected by SelectKBest:'
print features_selected

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

## Gaussian Classifier
from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()
g_clf.fit(features_train, labels_train)
g_pred = g_clf.predict(features_test)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(random_state=42)
a_clf.fit(features_train, labels_train)
a_pred = a_clf.predict(features_test)

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
r_clf = RandomForestClassifier(random_state=42)
r_clf.fit(features_train, labels_train)
r_pred = r_clf.predict(features_test)

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(features_train, labels_train)
dt_pred = dt_clf.predict(features_test)

## Evaluate Initial Classifiers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print "---------------------------------"
print "Precision score_Gaussian is:", precision_score(labels_test, g_pred)
print "---------------------------------"
print "Recall score_Gaussian is: ", recall_score(labels_test, g_pred)
print "---------------------------------"
print "---------------------------------"
print "Precision score_Adaboost is:", precision_score(labels_test, a_pred)
print "---------------------------------"
print "Recall score_Adaboost is: ", recall_score(labels_test, a_pred)
print "---------------------------------"
print "---------------------------------"
print "Precision score_RandomForest is:", precision_score(labels_test, r_pred)
print "---------------------------------"
print "Recall score_RandomForest is: ", recall_score(labels_test, r_pred)
print "---------------------------------"
print "---------------------------------"
print "Precision score_DecisionTree is:", precision_score(labels_test, dt_pred)
print "---------------------------------"
print "Recall score_DecisionTree is: ", recall_score(labels_test, dt_pred)
print "---------------------------------"


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# Example starting point. Try investigating other evaluation techniques!

sss = StratifiedShuffleSplit(n_splits= 100, test_size= 0.3, random_state= 42)
scaler = MinMaxScaler()
perc_var = .95
pca = PCA(n_components=perc_var)

clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
pipe = Pipeline(steps=[('scaler', scaler),('pca', pca),('skb', skb), ('gaussian', clf)])

params = dict(pca__n_components=[0.8,0.85,0.9,0.95],
                    skb__k = [1,2,3,4])

gs = GridSearchCV(pipe, params, cv=sss, scoring = 'f1')

##########
# Output
########## 

gs.fit(features_train, labels_train)
print "The best parameters for the grid:"
print gs.best_params_
print "---------------------------------"
print gs.best_score_
print "---------------------------------"
print ' '


clf = gs.best_estimator_
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print "---------------------------------"
print "Precision score is:", precision_score(labels_test, pred)
print "---------------------------------"
print "Recall score is: ", recall_score(labels_test, pred)
print "---------------------------------"
print "Classification Report:\n ", classification_report(labels_test, pred)
print "---------------------------------"
print "Confusion Matrix:\n ", confusion_matrix(labels_test, pred)
print "---------------------------------"



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)