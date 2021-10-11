#!/usr/bin/env python
# coding: utf-8

# ### Boston Housing Data
# 
# In order to gain a better understanding of the metrics used in regression settings, we will be looking at the Boston Housing dataset.  
# 
# First use the cell below to read in the dataset and set up the training and testing data that will be used for the rest of this problem.

# In[1]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import tests2 as t

boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)


# > **Step 1:** Before we get too far, let's do a quick check of the models that you can use in this situation given that you are working on a regression problem.  Use the dictionary and corresponding letters below to provide all the possible models you might choose to use.

# In[3]:


# When can you use the model - use each option as many times as necessary
a = 'regression'
b = 'classification'
c = 'both regression and classification'

models = {
    'decision trees': c, # Letter here,
    'random forest': c, # Letter here,
    'adaptive boosting': c, # Letter here,
    'logistic regression': b, # Letter here,
    'linear regression': a, # Letter here
}

#checks your answer, no need to change this code
t.q1_check(models)


# > **Step 2:** Now for each of the models you found in the previous question that can be used for regression problems, import them using sklearn.

# In[4]:


# Import models from sklearn - notice you will want to use 
# the regressor version (not classifier) - googling to find 
# each of these is what we all do!
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression


# > **Step 3:** Now that you have imported the 4 models that can be used for regression problems, instantate each below.

# In[5]:


# Instantiate each of the models you imported
# For now use the defaults for all the hyperparameters
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
ab = AdaBoostRegressor()
lr = LinearRegression()


# > **Step 4:** Fit each of your instantiated models on the training data.

# In[6]:


# Fit each of your models using the training data
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
ab.fit(X_train, y_train)
lr.fit(X_train, y_train)


# > **Step 5:** Use each of your models to predict on the test data.

# In[7]:


# Predict on the test values for each model
pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)
pred_ab = ab.predict(X_test)
pred_lr = lr.predict(X_test)


# > **Step 6:** Now for the information related to this lesson.  Use the dictionary to match the metrics that are used for regression and those that are for classification.

# In[12]:


# potential model options
a = 'regression'
b = 'classification'
c = 'both regression and classification'

#
metrics = {
    'precision': b, # Letter here,
    'recall': b, # Letter here,
    'accuracy': b, # Letter here,
    'r2_score': a, # Letter here,
    'mean_squared_error': a, # Letter here,
    'area_under_curve': b, # Letter here, 
    'mean_absolute_area': a, # Letter here 
}

#checks your answer, no need to change this code
t.q6_check(metrics)


# > **Step 6:** Now that you have identified the metrics that can be used in for regression problems, use sklearn to import them.

# In[15]:


# Import the metrics from sklearn
from sklearn.metrics import (
    r2_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    mean_squared_error, 
    mean_absolute_error,
    roc_auc_score,
)


# > **Step 7:** Similar to what you did with classification models, let's make sure you are comfortable with how exactly each of these metrics is being calculated.  We can then match the value to what sklearn provides.

# In[17]:


def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst

# Check solution matches sklearn
print(r2(y_test, pred_rf))
print(r2_score(y_test, pred_rf))
print("Since the above match, we can see that we have correctly calculated the r2 value.")


# > **Step 8:** Your turn fill in the function below and see if your result matches the built in for mean_squared_error. 

# In[20]:


def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''
    
    return np.sum((actual-preds)**2)/len(actual)


# Check your solution matches sklearn
print(mse(y_test, pred_rf))
print(mean_squared_error(y_test, pred_rf))
print("If the above match, you are all set!")


# > **Step 9:** Now one last time - complete the function related to mean absolute error.  Then check your function against the sklearn metric to assure they match. 

# In[22]:


def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''
    
    return np.sum(np.abs(actual-preds))/len(actual)

# Check your solution matches sklearn
print(mae(y_test, pred_rf))
print(mean_absolute_error(y_test, pred_rf))
print("If the above match, you are all set!")


# > **Step 10:** Which model performed the best in terms of each of the metrics?  Note that r2 and mse will always match, but the mae may give a different best model.  Use the dictionary and space below to match the best model via each metric.

# In[23]:


#match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'


best_fit = {
    'mse': b, # letter here,
    'r2': b, # letter here,
    'mae': b, # letter here
}

#Tests your answer - don't change this code
t.check_ten(best_fit)


# In[ ]:


# cells for work


# In[ ]:





# In[ ]:




