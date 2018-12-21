
# Tree Ensembles and Random Forests - Lab

## Introduction

In this lab, we'll create some popular Tree Ensemble models such as a Bag of Trees and a Random Forest to predict a person's salary based on information about them. 

## Objectives

You will be able to:

* Create, train, and make predictions with Bagging Classifiers
* Create, train, and make predictions with a Random Forest
* Understand and explain the concept of bagging as it applies to Ensemble Methods
* Understand and explain the Subspace Sampling Method and it's use in Random Forests

# Objectives 
YWBAT 
* Define a decision tree
    * **Classifier** that splits the data to maximize purity
* Define a random forest
    * **Ensemble** of Decision Trees
    * describe bagging **(short for bootstrap aggregration)** in a random forest
        * what is inbag/outofbag sampling
            * inbag -> 2/3
                * different for each tree
                * bootstrapped
            * outofbag -> 1/3
                * used for testing the tree by calculating error
        * why do we do this?
            * reduce variance
            * minimize error 
            * find the best tree
    * describe subspace sampling
        * why do we do this?
        * what is subspace sampling?
            * randomly sampling features
            * different features may have different predictive power
            * subspace sampling allows us to find useful features by making trees that without strong predictors
            * reduce variance
            * reduce bias

## Bootstrap sampling
* Sampling with replacement
    * ex: the numbers 1 - 30 and I want to bootstrap sample 30 points
    * [1, 1, 3, 3, 4, 4, 4, 10, 12, 12, 13, 14, 15, 15, 16, 16, 16, 16, 17, ...]

## 1. Importing the data

In this lab, we'll be looking at a dataset of information about people and trying to predict if they make more than 50k/year.  The salary data set was extracted from the census bureau database and contains salary information. The goal is to use this data set and to try to draw conclusions regarding what drives salaries. More specifically, the target variable is categorical (> 50k; <= 50 k). Let's create a classification tree!

To get started, run the cell below to import everything we'll need for this lab. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
```

Our dataset is stored in the file `salaries_final.csv`.  

In the cell below, read in the dataset from this file and store it in a DataFrame.  Be sure to set the `index_col` parameter to `0`.  Then, display the head of the DataFrame to ensure that everything loaded correctly.


```python
salaries = pd.read_csv("salaries_final.csv", index_col=0)
salaries.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education</th>
      <th>Occupation</th>
      <th>Relationship</th>
      <th>Race</th>
      <th>Sex</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>Bachelors</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Bachelors</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>HS-grad</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>11th</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Bachelors</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



In total, there are 6 predictors, and one outcome variable, the target salary <= 50k/ >50k.

recall that the 6 predictors are:

- `Age`: continuous.

- `Education`: Categorical. Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, 
Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

- `Occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

- `Relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

- `Race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

- `Sex`: Female, Male.

First, we'll need to store our `'Target'` column in a separate variable and drop it from the dataset.  

Do this in the cell below. 


```python
target = salaries.pop('Target')
target[:10]
```




    0    <=50K
    1    <=50K
    2    <=50K
    3    <=50K
    4    <=50K
    5    <=50K
    6    <=50K
    7     >50K
    8     >50K
    9     >50K
    Name: Target, dtype: object




```python
target_2 = [1 if t==">50K" else 0 for t in target]
target_2[:10]
```




    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]



Next, we'll want to confirm that the Age column is currently encoded in a numeric data type, and not a string. By default, pandas will treat all columns encoded as strings as categorical columns, and create a dummy column for each unique value contained within that column.  We do not want a separate column for each age, so let's double check that the age column is encoded as an integer or a float.  

In the cell below, check the `.dtypes` of the DataFrame to examine the data type of each column. 


```python
salaries.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32561 entries, 0 to 32560
    Data columns (total 6 columns):
    Age             32561 non-null int64
    Education       32561 non-null object
    Occupation      32561 non-null object
    Relationship    32561 non-null object
    Race            32561 non-null object
    Sex             32561 non-null object
    dtypes: int64(1), object(5)
    memory usage: 1.7+ MB



```python
salaries_dummies = pd.get_dummies(salaries, columns=["Education", "Occupation", "Relationship", "Race", "Sex"])
salaries_dummies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Education_10th</th>
      <th>Education_11th</th>
      <th>Education_12th</th>
      <th>Education_1st-4th</th>
      <th>Education_5th-6th</th>
      <th>Education_7th-8th</th>
      <th>Education_9th</th>
      <th>Education_Assoc-acdm</th>
      <th>Education_Assoc-voc</th>
      <th>...</th>
      <th>Relationship_Own-child</th>
      <th>Relationship_Unmarried</th>
      <th>Relationship_Wife</th>
      <th>Race_Amer-Indian-Eskimo</th>
      <th>Race_Asian-Pac-Islander</th>
      <th>Race_Black</th>
      <th>Race_Other</th>
      <th>Race_White</th>
      <th>Sex_Female</th>
      <th>Sex_Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>



Great.  Now we're ready to create some dummy columns and deal with our categorical variables.  

In the cell below, use pandas to create dummy columns for each of categorical variables.  If you're unsure of how to do this, check out the [documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html).  


```python
data = pd.get_dummies(salaries, columns=["Education", "Occupation", "Relationship", "Race", "Sex"])
```

Now, split your data and target into training and testing sets using the appropriate method from sklearn. 


```python
data_train, data_test, target_train, target_test = train_test_split(data, target_2, test_size = 0.25)
```


```python
data.shape
```




    (32561, 45)



## 2. Let's rebuild a "regular" tree as a baseline

We'll begin by fitting a regular Decision Tree Classifier, so that we have something to compare our ensemble methods to.  


```python
# Linear classifier
tree_clf = SGDClassifier()
tree_clf.fit(data_train, target_train)
tree_clf.score(data_test, target_test)
```

    /anaconda3/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)





    0.7870040535560742



### 2.1 Building the tree

In the cell below, create a Decision Tree Classifier.  Set the `criterion` to `'gini'`, and a `max_depth` of `5`.  Then, fit the tree to our training data and labels.  


```python
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
tree_clf.fit(data_train, target_train)
tree_clf.score(data_test, target_test)
```




    0.8188183269868566



### 2.1 Feature importance

Let's quickly examine how important each feature ended up being in our Decision Tree model.  Check the `feature_importances_` attribute of our trained model to see what it displays. 


```python
for feature, importance in zip(data.columns, tree_clf.feature_importances_):
    print("{} -> {}".format(feature, importance))
```

    Age -> 0.06717699680114476
    Education_10th -> 0.0
    Education_11th -> 0.0
    Education_12th -> 0.0
    Education_1st-4th -> 0.0
    Education_5th-6th -> 0.0
    Education_7th-8th -> 0.0
    Education_9th -> 0.0
    Education_Assoc-acdm -> 0.0
    Education_Assoc-voc -> 0.0
    Education_Bachelors -> 0.08918620532085557
    Education_Doctorate -> 0.0
    Education_HS-grad -> 0.0011260975093621175
    Education_Masters -> 0.013104175784617143
    Education_Preschool -> 0.0
    Education_Prof-school -> 0.005223214663212001
    Education_Some-college -> 0.0
    Occupation_? -> 0.0
    Occupation_Adm-clerical -> 0.0
    Occupation_Armed-Forces -> 0.0
    Occupation_Craft-repair -> 0.0
    Occupation_Exec-managerial -> 0.10001705092442034
    Occupation_Farming-fishing -> 0.0
    Occupation_Handlers-cleaners -> 0.0
    Occupation_Machine-op-inspct -> 0.0
    Occupation_Other-service -> 0.0
    Occupation_Priv-house-serv -> 0.0
    Occupation_Prof-specialty -> 0.10061359642756813
    Occupation_Protective-serv -> 0.0
    Occupation_Sales -> 0.0
    Occupation_Tech-support -> 0.0
    Occupation_Transport-moving -> 0.0
    Relationship_Husband -> 0.4997473169081841
    Relationship_Not-in-family -> 0.0
    Relationship_Other-relative -> 0.0
    Relationship_Own-child -> 0.0
    Relationship_Unmarried -> 0.0
    Relationship_Wife -> 0.1180728962955621
    Race_Amer-Indian-Eskimo -> 0.0
    Race_Asian-Pac-Islander -> 0.00028282412054154716
    Race_Black -> 0.0
    Race_Other -> 0.0
    Race_White -> 0.0
    Sex_Female -> 0.0
    Sex_Male -> 0.005449625244532264


That matrix isn't very helpful, but a visualization of the data it contains could be.  Run the cell below to plot a visualization of the feature importances for this model. Run the cell below to create a visualization of the data stored inside of a model's `.feature_importances_` attribute.


```python
def plot_feature_importances(model):
    n_features = data_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), data_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances(tree_clf)
```


![png](index_files/index_28_0.png)


### 2.3 Model performance

Next, let's see how well our model performed on the data. 

In the cell below:

* Use the classifier to create predictions on our test set. 
* Print out a `confusion_matrix` of our test set predictions.
* Print out a `classification_report` of our test set predictions.


```python
pred = tree_clf.predict(data_test)
print(confusion_matrix(target_test, pred))
print(classification_report(target_test, pred))
```

    [[5762  397]
     [1078  904]]
                 precision    recall  f1-score   support
    
              0       0.84      0.94      0.89      6159
              1       0.69      0.46      0.55      1982
    
    avg / total       0.81      0.82      0.80      8141
    



```python
target.value_counts()
```




    <=50K    24720
    >50K      7841
    Name: Target, dtype: int64



Now, let's check the model's accuracy. Run the cell below to display the test set accuracy of the model. 


```python
print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(target_test, pred) * 100))
```

    Testing Accuracy for Decision Tree Classifier: 81.88%


## 3. Bagged trees

The first Ensemble approach we'll try is a Bag of Trees.  This will make use of **_Bagging_**, along with a number of Decision Tree Classifier models.  

Now, let's create a `BaggingClassifier`.  In the first parameter spot, initialize a `DecisionTreeClassifier` and set the same parameters that we did above for `criterion` and `max_depth`.  Also set the `n_estimators` parameter for our Bagging Classifier to `20`. 


```python
bagged_tree = BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
```

Great! Now, fit it to our training data. 


```python
bagged_tree.fit(data_train, target_train)
```




    BaggingClassifier(base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'),
             bootstrap=True, bootstrap_features=False, max_features=1.0,
             max_samples=1.0, n_estimators=20, n_jobs=1, oob_score=False,
             random_state=None, verbose=0, warm_start=False)



Checking the accuracy of a model is such a common task that all (supervised learning) models contain a `score()` method that wraps the `accuracy_score` helper method we've been using.  All we have to do is pass it a dataset and the corresponding labels and it will return the accuracy score for those data/labels.  

Let's use it to get the training accuracy of our model. In the cell below, call the `.score()` method on our Bagging model and pass in our training data and training labels as parameters. 


```python
bagged_tree.score(data_train, target_train)
```




    0.8262899262899263



Now, let's check the accuracy score that really matters--our testing accuracy.  This time, pass in our testing data and labels to see how the model did.  


```python
bagged_tree.score(data_test, target_test)
```




    0.8236088932563567



## 4. Random forests

Another popular ensemble method is the **_Random Forest_** model.  Let's fit a Random Forest Classifier next and see how it measures up compared to all the others. 

### 4.1 Fitting a random forests model

In the cell below, create a `RandomForestClassifier`, and set the number estimators to `100` and the max depth to `5`. Then, fit the model to our training data. 


```python
forest = RandomForestClassifier(n_estimators=100, max_depth=5)
```

Now, let's check the training and testing accuracy of the model using its `.score()` method.


```python
forest.fit(data_train, target_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
forest.score(data_train, target_train)
```




    0.8058149058149058




```python
forest.score(data_test, target_test)
```




    0.8024812676575359



### 4.2 Look at the feature importances


```python
plot_feature_importances(forest)
```


![png](index_files/index_52_0.png)


Note: "relationship" represents what this individual is relative to others. For example an
individual could be a Husband. Each entry only has one relationship, so it is a bit of a weird attribute.

Also note that more features show up. This is a pretty typical result. 

### 4.3 Look at the trees in your forest

Let's create a forest with some small trees. You'll learn how to access trees in your forest!

In the cell below, create another `RandomForestClassifier`.  Set the number of estimators to 5, the `max_features` to 10, and the `max_depth` to 2.


```python
forest_2 = RandomForestClassifier(max_features=10, max_depth=2, n_estimators=5)

```


```python
forest_2.fit(data_train, target_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=2, max_features=10, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



Making `max_features` smaller will lead to very different trees in your forest!

The trees in your forest are stored in the `.estimators_` attribute.

In the cell below, get the first tree from `forest_2.estimators_` and store it in `rf_tree_1`


```python
rf_tree_1 = forest_2.estimators_[0]
```

Now, we can reuse ourn `plot_feature_importances` function to visualize which features this tree was given to use duing subspace sampling. 

In the cell below, call `plot_feature_importances` on `rf_tree_1`.


```python
plot_feature_importances(rf_tree_1)
```


![png](index_files/index_62_0.png)


Now, grab the second tree and store it in `rf_tree_2`, and then pass it to `plot_feature_importances` in the following cell so we can compare which features were most useful to each. 


```python
rf_tree_2 = forest_2.estimators_[1]
```


```python
plot_feature_importances(rf_tree_2)
```


![png](index_files/index_65_0.png)



```python
forest_2.score(data_test, target_test)
```




    0.7567866355484584



We can see by comparing the two plots that the two trees we examined from our Random Forest look at different attributes, and have wildly different importances for them!

## Summary

In this lab, we got some practice creating a few different Tree Ensemble Methods. We also learned how to visualize feature importances, and compared individual trees from a Random Forest to see if we could notice the differences in the features they were trained on. 


```python
bagged_forest = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1) # Bags by itself
```


```python
bagged_forest.fit(data_train, target_train)
bagged_forest.score(data_test, target_test)
```




    0.7940056504114974


