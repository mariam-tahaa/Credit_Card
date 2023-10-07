#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# # Reading data

# In[31]:


data = pd.read_csv("D:\\machine learning\\python_task\\credit_card_approval\\cc_approvals.data", header= None)
print(data.head())


# In[32]:


data_description = data.describe()
print(data_description)
print('\n')
data_info = data.info()
print(data_info)
print('\n')


# In[33]:


#Reading data from tail to shown missing values
data_tail= data.tail(17)
print(data_tail)


# # Splitting Data

# In[34]:


# Drop the features 11 and 13 
data = data.drop([11, 13], axis= 1)
# Split data
X= data.iloc[:, :-1]
y= data.iloc[:, -1]
X_train, X_test= train_test_split(data, test_size=0.33, random_state=42)


# # Clean Data

# In[39]:


# Replace the '?'s with NaN in the train and test sets
X_train = X_test.replace("?", np.NaN)
X_test = X_test.replace("?", np.NaN)


# In[40]:


# Impute the missing values with mean imputation
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

# Count the number of NaNs
print(X_train.isnull().sum().count())
print(X_test.isnull().sum().count())


# In[42]:


# Iterate over each column of data_train
for col in X_train.columns:
    # Check if the column is of object type
    if X_train[col].dtypes == 'object':
        X_train = X_train.fillna(X_train[col].value_counts().index[0])
        X_test = X_test.fillna(X_train[col].value_counts().index[0])

# Count the number of NaNs 
print(X_train.isnull().sum().count())
print(X_test.isnull().sum().count())


# # Preprocessing

# In[43]:


# Convert the categorical features in the train and test sets into numerical
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Reindex the columns of the test set aligning with the train set
X_test = X_test.reindex(columns= X_train.columns,fill_value= 0)


# # Scaling data

# In[44]:


from sklearn.preprocessing import MinMaxScaler
X_train, y_train = X_train.iloc[: , :-1].values, X_train.iloc[:, -1].values
X_test, y_test = X_test.iloc[: , :-1].values, X_test.iloc[:, -1].values

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(copy= True, feature_range= (0,1))
rescaledX_train = scaler.fit_transform(X_train, y_train)
rescaledX_test = scaler.transform(X_test)


# # Perform the model

# In[57]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(rescaledX_train, y_train)


# # Test performance

# In[58]:


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of model 
print("Accuracy of classifier: ", logreg.score(rescaledX_test, y_test))
confusion_matrix(y_test, y_pred)


# ### Additional steps in case we need to increase performance

# In[59]:


from sklearn.model_selection import GridSearchCV
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict({'tol': [0.01, 0.001, 0.0001], 'max_iter': [100, 150, 200]})


# ### Getting bast performing model

# In[60]:


grid_model = GridSearchCV(estimator= regressor, param_grid= param_grid, cv= 5)

# Fit grid_model to the data
grid_model_result = grid_model.fit(rescaledX_train, y_train)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("Accuracy of logistic regression classifier: ", best_model.score(rescaledX_train, y_train))

