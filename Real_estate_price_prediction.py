#!/usr/bin/env python
# coding: utf-8

# ## *A real estate company wants to predict the prices for the future based on the data available. This way it will help them adjust their budget, change the way they work so as to earn profit and also help them decide where to invest in. As a Data Scientist, I will explore the data available, perform preprocessing, check for manipulation and finally apply and evaluate machine learning algorithms to see which algorithm provides the best result.*

# ## Reading the data

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt


# In[2]:


housing_data = pd.read_csv("data.csv")


# In[3]:


housing_data.head()


# # Attribute description
#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per $10,000
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in $1000's

# In[4]:


housing_data.info()


# In[5]:


housing_data.describe()


# In[6]:


housing_data.hist(bins=50,figsize=(20,15))


# ## Splitting using manual method(numpy)

# In[7]:


def splitting(data,test_ratio):
    np.random.seed(123)
    ratio = np.random.permutation(len(data))
    print(ratio)
    test_size = int(len(data)*test_ratio)
    test_indices = ratio[:test_size]
    train_indices = ratio[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train, test = splitting(housing_data,0.2)


# In[9]:


print(f"The number of rows in training dataset is: {len(train)}\nThe number of rows in test dataset is: {len(test)}")


# ## Splitting using sklearn

# In[10]:


from sklearn.model_selection import train_test_split
train_data , test_data = train_test_split(housing_data, test_size = 0.2, random_state = 121)
print(f"The number of rows in training dataset is: {len(train_data)}\nThe number of rows in test dataset is: {len(test_data)}")


# We need to make sure that the values in 'CHAS' variable is appropriately distributed in both test and train dataset. For this, I will make use of stratified shuffling from sklearn package which will make sure that all possible values are included in both train and test

# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 121)
for train_index, test_index in sss.split(housing_data, housing_data['CHAS']):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


strat_train_set['CHAS'].value_counts()


# In[14]:


#copying the training dataset to main dataset as we need to work with the training data
housing_data = strat_train_set.copy()


# ## Correlation

# In[15]:


corr_matrix = housing_data.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[16]:


from pandas.plotting import scatter_matrix
attr = ["MEDV", "RM","ZN","LSTAT"]
scatter_matrix(housing_data[attr], figsize=(12,8))


# In[17]:


housing_data.plot(kind = "scatter", x = "RM", y="MEDV", alpha=0.8)


# In[18]:


housing_data["TAX_PER_ROOM"] = housing_data["TAX"]/housing_data["RM"]
housing_data["TAX_PER_ROOM"]


# In[19]:


corr_matrix = housing_data.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


housing_data.plot(kind = "scatter", x = "TAX_PER_ROOM", y="MEDV", alpha=0.8)


# ## Handling Missing data
# 
# 1. Remove the missing data points
# 2. Remove the whole attribute/column/row
# 3. Set some value(0,mean or median)

# In[21]:


housing_label = strat_train_set["MEDV"].copy()
housing_data = strat_train_set.drop("MEDV", axis = 1)


# In[22]:


#calculating number if NA values in each attributes
housing_data.isna().sum()


# In[23]:


#Since there is a strong correlation, we cannot drop the column. However we will see how every step is executed if need be

#Option 1

x = housing_data.dropna(subset = ['RM'])
x.shape


# In[24]:


#option 2

y = housing_data.drop('RM', axis = 1)
y.shape


# In[25]:


#option 3

z = housing_data['RM'].fillna(housing_data['RM'].median()) #can be mean or 0 as well
z


# In[26]:


housing_data.describe()


# # making the 3rd option automated using sklearn

# In[27]:


from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy = "median")

#this will work for all the attributes in the dataset - replace all the NA values in any column as its column's median value
impute.fit(housing_data)


# In[28]:


#median of each attribute after imputing
impute.statistics_


# In[29]:


X_impute = impute.transform(housing_data)


# In[30]:


housing_data_tr = pd.DataFrame(X_impute, columns = housing_data.columns)
housing_data_tr.describe()


# # Designing Sklearn
# 
# The objects in scikit learn are -
# 1. Estimators - It is used to estimate something provided there are some conditions. Imputation is a good example of estimators. 
# This includes a fit and the transform method where fit method fits the dataset with some conditions and calculate the internal parameters. 
# 
# 2. Transformers - It learns from the fit method and calculates a task by taking input values. It has a function "fit_transform" which will fit and transform in a single step
# 
# 3. Predictors - LinearRegression has fit() and predict() method along with the accuracy score of the model.

# ## Feature scaling
# This is used to bring out one common scale for every interger value/column
# 
# 1. min-max or Normalisation method
# 
# (value - min)/(max - min) The range would be from 0 to 1.
# sklearn has MinMaxScaler to perform this step
# 
# 2. Standardization
# 
# (value - mean)/SD
# The result would have a variance of 1 and mean would be equal to the original mean.
# sklearn has StandardScaler to perform this step
# 

# ## Creating pipeline
# 
# Right from imputing with median values, feature scaling to machine learning, everything is automated using pipeline

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline_data = Pipeline([
    ('Imputer', SimpleImputer(strategy = "median")),
    ('std_scaler', StandardScaler())
])


# In[40]:


#converts to numpy array
housing_tr = pipeline_data.fit_transform(housing_data)


# In[41]:


#The above step just took the data we fed, started with the imputing and then standardized it with standardscaler
housing_tr


# In[42]:


housing_tr.shape


# # Model Selection

# ### Linear Regression

# In[43]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(housing_tr, housing_label)


# In[47]:


some_data = housing_data.iloc[:5]
some_labels = housing_label.iloc[:5]

data_prepared = pipeline_data.transform(some_data)
lr.predict(data_prepared)


# In[107]:


some_labels


# ## Model Evaluation for Linear Regression

# In[52]:


from sklearn.metrics import mean_squared_error
predicted_data = lr.predict(housing_tr)
lr_mse = mean_squared_error(housing_label, predicted_data)
lr_RMSE = np.sqrt(lr_mse)


# In[54]:


lr_RMSE


# There a small amount of root mean squared error which normally would be acceptable. We will check for other algorithms as well

# ### Decision Trees

# In[56]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(housing_tr,housing_label)


# In[58]:


dtr.predict(data_prepared)


# In[61]:


some_labels


# ## Model Evaluation for Decision Trees

# In[64]:


predicted_data_dtr = dtr.predict(housing_tr)
dtr_mse = mean_squared_error(housing_label, predicted_data_dtr)
dtr_RMSE = np.sqrt(dtr_mse)


# In[66]:


dtr_RMSE


# The model showed 0 root mean squared error. This is an example of overfitting. The model basically learned the noise too to fit it well which is not good

# ### Random Forest

# In[85]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(housing_tr, housing_label)


# In[86]:


rfr.predict(data_prepared)


# In[87]:


some_labels


# ## Model Evaluation for Random Forest

# In[88]:


predicted_data_rfr = rfr.predict(housing_tr)
rfr_mse = mean_squared_error(housing_label, predicted_data_rfr)
rfr_RMSE = np.sqrt(rfr_mse)


# In[89]:


rfr_RMSE


# ## Cross Validation

# **For Decision Trees**

# In[75]:


from sklearn.model_selection import cross_val_score

#to increase the utility, we will use scoring as neg_mean_squared_error
cvs_dtr = cross_val_score(dtr, housing_tr, housing_label, scoring = "neg_mean_squared_error", cv = 10)
cv_rmse = np.sqrt(-cvs_dtr)


# In[76]:


cv_rmse


# **For Linear Regression**

# In[79]:


cvs_lr = cross_val_score(lr, housing_tr, housing_label, scoring = "neg_mean_squared_error", cv = 10)
cv_lr_rmse = np.sqrt(-cvs_lr)


# In[81]:


cv_lr_rmse


# **For Random Forest**

# In[91]:


cvs_rfr = cross_val_score(rfr, housing_tr, housing_label, scoring = "neg_mean_squared_error", cv = 10)
cv_rfr_rmse = np.sqrt(-cvs_rfr)


# In[92]:


cv_rfr_rmse


# #### Function to print rmse scores, mean and standard deviation

# In[82]:


def details(score):
    print("RMSE: ", score)
    print("Mean: ", score.mean())
    print("Standard deviation: ", score.std())


# In[83]:


#For decision tree
details(cv_rmse)


# In[84]:


#For Linear regression
details(cv_lr_rmse)


# In[93]:


#For Random forest
details(cv_rfr_rmse)


# **We can clearly notice that the Random Forest worked very well as compared to other models. We will consider random forest for our final usage on the whole dataset**

# In[105]:


from joblib import dump, load
dump(rfr, 'Model_use.joblib')


# # Model Testing on test dataset

# In[94]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()


# **Using Linear regression model**

# In[98]:


X_test_prepared_lr = pipeline_data.transform(X_test)
lr_predictions = lr.predict(X_test_prepared_lr)
test_lr_mse = mean_squared_error(Y_test, lr_predictions)
test_lr_rmse = np.sqrt(test_lr_mse)
test_lr_rmse


# In[102]:


values_lr = pd.DataFrame({"lr_predictions":lr_predictions,"Y_test":Y_test, "Residual": Y_test-lr_predictions})
values_lr.head()


# **Using Decision tree model**

# In[96]:


X_test_prepared_dtr = pipeline_data.transform(X_test)
dtr_predictions = dtr.predict(X_test_prepared_dtr)
test_dtr_mse = mean_squared_error(Y_test, dtr_predictions)
test_dtr_rmse = np.sqrt(test_dtr_mse)
test_dtr_rmse


# In[103]:


values_dtr = pd.DataFrame({"dtr_predictions":dtr_predictions,"Y_test":Y_test, "Residual": Y_test-dtr_predictions})
values_dtr.head()


# **Using Random Forest Regressor**

# In[97]:


X_test_prepared_rfr = pipeline_data.transform(X_test)
rfr_predictions = rfr.predict(X_test_prepared_rfr)
test_rfr_mse = mean_squared_error(Y_test, rfr_predictions)
test_rfr_rmse = np.sqrt(test_rfr_mse)
test_rfr_rmse


# In[104]:


values_rfr = pd.DataFrame({"rfr_predictions":rfr_predictions,"Y_test":Y_test, "Residual": Y_test-rfr_predictions})
values_rfr.head()


# ## Model Usage

# In[112]:


#Here, random forest has been considered as the chosen model
from joblib import dump, load
model_use = load('Model_use.joblib')

features = [data_prepared[0]]
model_use.predict(features)

