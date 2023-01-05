#!/usr/bin/env python
# coding: utf-8

# ## 2011 MLB Pitch Prediction 
# 
# Lina Nguyen

# ### Importing all packages and data

# In[1]:


# EDA, data manipulation, and cleaning packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

# fixing imbalance
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# modeling
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from dmba import classificationSummary, gainsChart

# model evaluation
from sklearn.metrics import classification_report

classes = ('Satisfied', 'Unsatisfied')

def confusionMatrices(model, title):
    print(title + ' - training results')
    classificationSummary(y_train, model.predict(X_train), class_names=classes)
    print(title + ' - testing results')
    y_pred = model.predict(X_test)
    classificationSummary(y_test, y_pred, class_names=classes)
    
# model optimization
from sklearn.model_selection import GridSearchCV


# In[2]:


# import data
df = pd.read_csv('pitches', sep = ',')


# ### Feature Engineering 

# In[3]:


df.shape


# In[4]:


# view missing values
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
pd.set_option('display.max_rows', None)
missing_value_df.head(125)


# Columns with 100% missing values will be dropped from the dataset. event2, event3, and event4 will also be dropped because of the significant amount of missing values.

# In[5]:


perc = 98
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
mod_df = df.dropna(axis=1, 
                thresh=min_count)


# In[6]:


# Viewing number of unique values
mod_df.nunique().sort_values(ascending = False)


# - Columns with only 1 unique value, as well as a lot of unique categorical values (like id's) will be dropped because is lack of useful information. 
# - created_at, added_at, modified_at, modified_by will be dropped because it is irrelavant to the pitcher. 
# - Columns with descriptive text will also be dropped. 
# - type gets good representation in pitch_des, and will be dropped.
# - Dropping datetime timestamps columns because there are too many different values, and specific times might make the model bias. The inning column should be proficient enough.
# - Physics of movement in the ball, especially if the information is not available prior to the pitch, will also be dropped because the pitcher does not necessarily use the exact specific numbers to determine the next pitch. 
# - Final game statistics also don't play a role in a pitcher's in game decision, and will be dropped.

# In[7]:


mod_df = mod_df.drop(['y0','uid', 'created_at', 'added_at', 'modified_at', 'modified_by', 'pitch_id', 
                      'sv_id', 'game_pk', 'year', 'cc', 'at_bat_des', 'type',
                     'pitch_tfs', 'pitch_tfs_zulu', 'start_tfs', 'start_tfs_zulu', 'date',
                     'x', 'y', 'start_speed', 'end_speed', 'sz_top', 'sz_bot', 'pfx_x', 'pfx_z',
                     'px', 'pz', 'x0', 'z0', 'vx0', 'vz0', 'vy0', 'ax', 'az', 'ay', 'break_length', 
                     'break_y', 'break_angle','type_confidence', 'zone', 'nasty', 'spin_dir', 
                     'spin_rate', 'is_final_pitch', 'final_balls', 'final_strikes', 'final_outs'], axis = 1)


# score, on_1b, on_2b, on_3b will be represented by binary columns. Missing values will be represented as 0, and the rest will be represented as 1.

# In[8]:


# changing score, on_1b, on_2b, on_3b to binary columns
mod_df[['score', 'on_1b', 'on_2b', 'on_3b']] = mod_df[['score', 'on_1b', 'on_2b', 'on_3b']].where(mod_df[['score', 'on_1b', 'on_2b', 'on_3b']].isnull(), 1).fillna(0).astype(int)


# b_height is converted from datetime into inches, to create a new column, b_height_in

# In[9]:


# convert b_height from datetime to inches
mod_df['b_height'].head()

mod_df[["ft", "inches"]] = mod_df["b_height"].str.split("-", expand = True)

mod_df = mod_df.drop(['b_height'], axis = 1)

# convert ft*12 and inches from object to int, and add both columns to create b_height_in

mod_df[['ft', 'inches']] = mod_df[['ft', 'inches']].astype(str).astype(int)

mod_df['b_height_in'] = mod_df['ft']*12 + mod_df['inches']

mod_df = mod_df.drop(['ft', 'inches'], axis = 1)


# pitch_type will be compressed based on these [pitching acronyms.](https://library.fangraphs.com/pitch-type-abbreviations-classifications/#:~:text=Four%2Dseam%20fastballs%20are%20considered,act%20in%20a%20similar%20manner.)
# In summary, SI and FS both represent a fast ball, and PO and FO both represent a pitch out.

# In[10]:


mod_df['pitch_type'] = mod_df['pitch_type'].replace('FS', 'SI')
mod_df['pitch_type'] = mod_df['pitch_type'].replace('PO', 'FO')


# The missing data from the target column, pitch_type, is only missing 0.3%, the null values will be dropped. Since it is our target variable and not that much data is missing, trying to impute may end up training a model that is less accurate at predicting the target variable.

# In[11]:


# unidentified pitches, UN, will be replaced with NaN, and all NaN rows will be dropped
mod_df['pitch_type'] = mod_df['pitch_type'].replace('UN', np.nan)

# drop columns with null values for pitch_type
mod_df = mod_df[mod_df['pitch_type'].notna()]

mod_df.shape


# In[12]:


mod_df.dtypes


# Checking for multicolinearity.

# In[13]:


y = mod_df['pitch_type']
mod_df = mod_df.drop(['pitch_type'], axis = 1)


# In[14]:


# break into pred dataset to get dummy variables
mod_df = pd.get_dummies(data = mod_df, columns = ['stand', 'p_throws', 'pitch_des', 'event'], drop_first = True)


# In[15]:


plt.figure(figsize = (70,70))
sns.heatmap(mod_df.corr(), annot=True)


# Variables with a correlation coefficient of 0.7 or more will be dropped, and variables around 0.6 will be considered before dropping. 
# - We can see multicollineartiy between balls/strikes and pcount_at_bat, and inning and at_bat_num. pcount_at_bat and at_bat_num will be dropped.
# - event_Home Run will be dropped due to correlation with score
# - event_Hit By Pitch will be dropped due to correlation with pitch_des_Hit By Pitch
# - event_Walk will be dropped due to correlation with pitch_des_Ball

# In[16]:


mod_df = mod_df.drop(['pcount_at_bat', 'at_bat_num','event_Home Run','event_Hit By Pitch', 'event_Walk'], axis = 1) 


# Checking for missing values.

# In[17]:


mod_df.isna().sum()


# Columns with 0.3% missing values no longer need to be imputed because there are no more na values in the dataset. This is because rows with a missing target value were dropped earlier.

# ### EDA
# 
# Interactive EDA can be found [here.](http://192.168.1.42:8501)

# In[18]:


mod_df['pitch_type'] = y


# In[19]:


# pitch distribution
fig = plt.figure(figsize = (10,5))
ax = mod_df['pitch_type'].value_counts().plot(kind = 'barh', title = 'Pitch Frequency')
ax.set_ylabel('Pitch Label')
ax.set_xlabel('Frequency')


# There is clear class imbalance.  

# In[20]:


mod_df['pitch_type'].value_counts(normalize = True)*100


# KC, KN, IN, FO, FA, EP, SC, AB collectively appear in less than 2% of the entire dataset. These classes will be removed because they have too few instances. 

# In[21]:


def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

mod_df = filter_rows_by_values(mod_df, "pitch_type", ['AB', 'SC', 'EP', 'FA', 'FO', 'IN', 'KN', 'KC'])


# In[22]:


mod_df.shape


# ### Preparing the Dataset for Modeling

# In[23]:


# resetting index
mod_df = mod_df.reset_index()

# scale data and split into test and train
X = mod_df.drop(['pitch_type'], axis = 1)

# convert pitch_type into numerical representations
pitch_label = {'FF': 0, 'SL':1, 'CU':2, 'SI':3, 'FC':4, 'FT':5, 'CH':6}
y = mod_df['pitch_type'].map(pitch_label)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SMOTE (Synthetic Minority Over-sampling Technique) will be used to address the class imbalance. 

# In[24]:


oversample = SMOTE(random_state = 42)
X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[25]:


counter = Counter(y_train)
for k,v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution


# ### Modeling

# #### XGBoost 

# In[26]:


xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
xg_confusion = confusionMatrices(xg, 'XG Boost')


# In[27]:


print(classification_report(y_test, xg.predict(X_test)))


# #### Decision Tree 

# In[28]:


dtree = DecisionTreeClassifier().fit(X_train, y_train)
tree_confusion = confusionMatrices(dtree, 'Decision Tree')


# In[29]:


print(classification_report(y_test, dtree.predict(X_test)))


# #### Random Forest

# In[30]:


rf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train.values.ravel())
forest_confusion = confusionMatrices(rf, 'Random Forest')


# In[31]:


print(classification_report(y_test, rf.predict(X_test)))


# #### Bagging

# In[32]:


#we use the decision classification tree as the base estimator
bagging = BaggingClassifier(dtree, max_samples = 0.5, max_features = 0.5)
bagging.fit(X_train, y_train)
bag_confusion = confusionMatrices(bagging, 'Bagging')


# In[33]:


print(classification_report(y_test, bagging.predict(X_test)))


# #### Adaboost 

# In[34]:


adaboost = AdaBoostClassifier(n_estimators = 10, base_estimator = dtree)
adaboost.fit(X_train, y_train)
ada_confusion = confusionMatrices(adaboost, 'Adaboost')


# In[35]:


print(classification_report(y_test, adaboost.predict(X_test)))


# All models except for xgboost are overfit, and have low performance. The xgboost algorithm classifies the testing dataset best, but overall has very low performance metrics. XGBoost is the only algorithm that isn't overfit, but does not do a good job of classifing the training data. Hyperparameters will be tuned for xgboost to try to improve performance metrics, and random forest to reduce overfitting. 

# ### Model Optimization with GridSearchCV
# 
# Model optimization would not run due to limitations of technology.

# #### XGBoost 

# In[ ]:


parameters = {
    'max_depth': range (2, 10, 1), 'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]}
grid_search = GridSearchCV(estimator=xg, param_grid=parameters,
            scoring = 'roc_auc', n_jobs = 10, cv = 10, verbose=True)
grid_search.fit(X, y)
grid_search.best_estimator_


# #### Random Forest

# In[ ]:


param_grid = {
    'base_estimator__max_depth' : [1, 2, 3, 4, 5],
    'max_samples' : [0.05, 0.1, 0.2, 0.5]
}

clf = GridSearchCV(BaggingClassifier(DecisionTreeClassifier,
                                     n_estimators = 100, max_features = 0.5),
                   param_grid, scoring = choosen_scoring)
dtreeop = clf.fit(X_train, y_train)
dtreeop.best_estimator
optree_confusion = confusionMatrices(dtreeop, 'Decision Tree')
print(classification_report(y_test, rf.predict(X_test)))


# ### Probability of Pitch Type Using XGBoost and Random Forest

# #### XGBoost

# In[43]:


print('The next 5 predicted pitches are', xg.predict(X_test[:5]))


# In[44]:


xgprob = xg.predict_proba(X_test)[:5]
xgprob = pd.DataFrame(data = xgprob)
xgprob.columns = ['FF', 'SL', 'CU', 'SI', 'FC', 'FT', 'CH']
print(xgprob)


# XGBoost predicts the next 5 pitches are SL, SL, FF, FF, and FF.

# #### Random Forest 

# In[45]:


print('The next 5 predicted pitches are', rf.predict(X_test[:5]))


# In[47]:


rfprob = rf.predict_proba(X_test)[:5]
rfprob = pd.DataFrame(data = rfprob)
rfprob.columns = ['FF', 'SL', 'CU', 'SI', 'FC', 'FT', 'CH']
print(rfprob)


# Random Forest predicts the next five pitches are SL, SL, SL, FT, and FT.

# ### Conclusion and Future Work

# Due to major class imabalance, all the models had trouble with predicting the test set, even after balancing the training sets. During this project, I trained each model with different datasets, the first one being the original unbalanced dataset, which obviously performed very well. The second data set, I tried to use a combination of undersampling and oversampling to train the models, however, it ended up with the lowest metrics, probably because of the lost data from underfitting. The third dataset, I overfit the entire dataset. Not only did it take forever for my models to train, it also resulted in poor performance, because of the inclusion of all the classifiers. My last dataset, I decided to drop the less frequent classifiers to help boost SMOTE, and it resulted in the highest performing model of the 4 sets. 
# 
# The next steps I would take to improve the model is to use feature importance to help reduce overfitting in the model, and also test to see if furthermore reducing the amount of less frequent classifiers will boost metrics. I would also ideally have a more powerful computer so that I wouldn't have to wait ages for my models to run, be able to run GridSearchCV to further tune the hyperparameters, and run more models to see if they are better at performing this classification problem. 
# 
# As of now, I would say that xgboost and random forest performed best out of all the models, but further tweaking of the feature engineering and hyperparameters will help me determine which one is better.
