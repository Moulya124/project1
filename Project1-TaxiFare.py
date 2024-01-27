#!/usr/bin/env python
# coding: utf-8

# # NYC Taxi Fare Prediction

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import lightgbm as lgbm
import xgboost as xgb

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 4000000)
test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
df.shape, test_df.shape


# In[3]:


df.head()


# In[4]:


df.isnull().sum().sort_index()/len(df)


# In[5]:


df.dropna(subset=['dropoff_latitude', 'dropoff_longitude'], inplace = True)


# In[6]:


df.describe()


# In[7]:


df.drop(df[df['fare_amount'] < 2.5].index, axis=0, inplace = True)
df.drop(df[df['fare_amount'] > 500].index, axis=0, inplace = True)


# In[8]:


test_df.describe()


# In[9]:


df[df['passenger_count'] > 5].sort_values('passenger_count')


# In[10]:


df.drop(df[df['pickup_longitude'] == 0].index, axis=0, inplace = True)
df.drop(df[df['pickup_latitude'] == 0].index, axis=0, inplace = True)
df.drop(df[df['dropoff_longitude'] == 0].index, axis=0, inplace = True)
df.drop(df[df['dropoff_latitude'] == 0].index, axis=0, inplace = True)
df.drop(df[df['passenger_count'] == 208].index, axis=0, inplace = True)
df.drop(df[df['passenger_count'] > 5].index, axis=0, inplace = True)
df.drop(df[df['passenger_count'] == 0].index, axis=0, inplace = True)


# In[11]:


df['key'] = pd.to_datetime(df['key'])
key = test_df.key
test_df['key'] = pd.to_datetime(test_df['key'])
df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
test_df['pickup_datetime']  = pd.to_datetime(test_df['pickup_datetime'])


# In[12]:


df['Year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Date'] = df['pickup_datetime'].dt.day
df['Day of Week'] = df['pickup_datetime'].dt.dayofweek
df['Hour'] = df['pickup_datetime'].dt.hour
df.drop('pickup_datetime', axis = 1, inplace = True)
df.drop('key', axis = 1, inplace = True)

test_df['Year'] = test_df['pickup_datetime'].dt.year
test_df['Month'] = test_df['pickup_datetime'].dt.month
test_df['Date'] = test_df['pickup_datetime'].dt.day
test_df['Day of Week'] = test_df['pickup_datetime'].dt.dayofweek
test_df['Hour'] = test_df['pickup_datetime'].dt.hour
test_df.drop('pickup_datetime', axis = 1, inplace = True)
test_df.drop('key', axis = 1, inplace = True)


# In[13]:


df.dropna(inplace=True)

df.drop(df.index[(df.pickup_longitude < -75) | 
           (df.pickup_longitude > -72) | 
           (df.pickup_latitude < 40) | 
           (df.pickup_latitude > 42)],inplace=True)
df.drop(df.index[(df.dropoff_longitude < -75) | 
           (df.dropoff_longitude > -72) | 
           (df.dropoff_latitude < 40) | 
           (df.dropoff_latitude > 42)],inplace=True)


# In[14]:


df.describe()


# In[15]:


import geopy.distance

def geodesic_dist(trip):
    pickup_lat = trip['pickup_latitude']
    pickup_long = trip['pickup_longitude']
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    distance = geopy.distance.geodesic((pickup_lat, pickup_long), 
                                       (dropoff_lat, dropoff_long)).miles
    try:
        return distance
    except ValueError:
        return np.nan
    
def circle_dist(trip):
    pickup_lat = trip['pickup_latitude']
    pickup_long = trip['pickup_longitude']
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    distance = geopy.distance.great_circle((pickup_lat, pickup_long), 
                                       (dropoff_lat, dropoff_long)).miles
    try:
        return distance
    except ValueError:
        return np.nan


# In[16]:


def jfk_dist(trip):
    jfk_lat = 40.6413
    jfk_long = -73.7781
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    jfk_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (jfk_lat, jfk_long)).miles
    return jfk_distance

def lga_dist(trip):
    lga_lat = 40.7769
    lga_long = -73.8740
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    lga_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (lga_lat, lga_long)).miles
    return lga_distance

def ewr_dist(trip):
    ewr_lat = 40.6895
    ewr_long = -74.1745
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    ewr_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (ewr_lat, ewr_long)).miles
    return ewr_distance

def tsq_dist(trip):
    tsq_lat = 40.7580
    tsq_long = -73.9855
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    tsq_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (tsq_lat, tsq_long)).miles
    return tsq_distance

def cpk_dist(trip):
    cpk_lat = 40.7812
    cpk_long = -73.9665
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    cpk_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (cpk_lat, cpk_long)).miles
    return cpk_distance

def lib_dist(trip):
    lib_lat = 40.6892
    lib_long = -74.0445
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    lib_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (lib_lat, lib_long)).miles
    return lib_distance

def gct_dist(trip):
    gct_lat = 40.7527
    gct_long = -73.9772
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    gct_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (gct_lat, gct_long)).miles
    return gct_distance

def met_dist(trip):
    met_lat = 40.7794
    met_long = -73.9632
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    met_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (met_lat, met_long)).miles
    return met_distance

def wtc_dist(trip):
    wtc_lat = 40.7126
    wtc_long = -74.0099
    dropoff_lat = trip['dropoff_latitude']
    dropoff_long = trip['dropoff_longitude']
    wtc_distance = geopy.distance.geodesic((dropoff_lat, dropoff_long), (wtc_lat, wtc_long)).miles
    return wtc_distance


# In[17]:


def optimize_floats(df):
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df):
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize(df):
    return optimize_floats(optimize_ints(df))


# In[18]:


df = optimize(df)
test_df = optimize(test_df)


# In[19]:


def calc_dists(df):
    df['geodesic'] = df.apply(lambda x: geodesic_dist(x), axis = 1 )
    df['circle'] = df.apply(lambda x: circle_dist(x), axis = 1 )
    df['jfk'] = df.apply(lambda x: jfk_dist(x), axis = 1 )
    df['lga'] = df.apply(lambda x: lga_dist(x), axis = 1 )
    df['ewr'] = df.apply(lambda x: ewr_dist(x), axis = 1 )
    df['tsq'] = df.apply(lambda x: tsq_dist(x), axis = 1 )
    df['cpk'] = df.apply(lambda x: cpk_dist(x), axis = 1 )
    df['lib'] = df.apply(lambda x: lib_dist(x), axis = 1 )
    df['gct'] = df.apply(lambda x: gct_dist(x), axis = 1 )
    df['met'] = df.apply(lambda x: met_dist(x), axis = 1 )
    df['wtc'] = df.apply(lambda x: wtc_dist(x), axis = 1 )
    return df


# In[20]:


df = calc_dists(df)
test_df = calc_dists(test_df)


# In[21]:


# df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)
# test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis=1,inplace=True)


# In[22]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.drop('fare_amount', axis=1).corr(), square=True)
plt.suptitle('Pearson Correlation Heatmap')
plt.show();


# In[23]:


(mu, sigma) = norm.fit(df['geodesic'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))
ax1 = sns.distplot(df['geodesic'] , fit=norm, ax=ax1)
ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')
ax1.set_ylabel('Frequency')
ax1.set_title('Distance Distribution')
ax2 = stats.probplot(df['geodesic'], plot=plt)
f.show();


# In[24]:


(mu, sigma) = norm.fit(df['fare_amount'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))
ax1 = sns.distplot(df['fare_amount'] , fit=norm, ax=ax1)
ax1.legend([f'Normal distribution ($\mu=$ {mu:.3f} and $\sigma=$ {sigma:.3f})'], loc='best')
ax1.set_ylabel('Frequency')
ax1.set_title('Fare Distribution')
ax2 = stats.probplot(df['fare_amount'], plot=plt)
f.show();


# In[25]:


df.describe()


# In[26]:


df = optimize(df)
test_df = optimize(test_df)


# In[27]:


df.dtypes


# In[28]:


X, y = df.drop('fare_amount', axis = 1), df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# In[29]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dtest = xgb.DMatrix(test_df)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_params = {
    'min_child_weight': 1, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.7, 
    'max_depth': 10,
    'subsample': 0.7,
    'n_estimators': 5000,
    'n_jobs': -1, 
    'booster' : 'gbtree', 
    'silent': 1,
    'eval_metric': 'rmse'}

model = xgb.train(xgb_params, dtrain, 700, watchlist, early_stopping_rounds=100, maximize=False, verbose_eval=50)


# In[30]:


y_train_pred = model.predict(dtrain)
y_pred = model.predict(dvalid)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')


# In[31]:


test_preds = model.predict(dtest)


# In[32]:


test_preds = model.predict(dtest)

submission = pd.DataFrame(
    {'key': key, 'fare_amount': test_preds},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission1.csv', index = False)


# In[ ]:




