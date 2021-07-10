#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt

data = pd.read_csv(r'./../UserBehavior.csv',header=None, names=['user_ID','item_ID','category_ID','behavior_type','time_stamp'])
data


# In[2]:


data = data.sort_values(by = ['user_ID','time_stamp'])
data


# In[3]:


data = data[data['behavior_type'] == 'pv']
data


# In[4]:


del(data['behavior_type'])


# In[5]:


data = data[data['time_stamp'] >= 1511539200] # 2017/11/25 00:00:00
data = data[data['time_stamp'] < 1512316800] # 2017/12/4 00:00:00
data


# In[6]:


data.describe()


# In[7]:


data['time_stamp_diff'] = data.groupby('user_ID')['time_stamp'].apply(lambda x: x.diff(1))
data


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
sns.kdeplot(data['time_stamp_diff'])


# In[8]:


for i in np.arange(0.01,1,0.01):
    print("%f %f" %(i, data['time_stamp_diff'].quantile(i)))


# In[8]:


session_gap_threshold = data['time_stamp_diff'].quantile(0.8)
data['if_new_session'] = np.isnan(data['time_stamp_diff']) | (data['time_stamp_diff'] > session_gap_threshold) # use '|' instead of 'or' here
data['session_ID'] = data['if_new_session'].astype('int').cumsum()
data


# In[19]:


'''
session_gap_threshold = data['time_stamp_diff'].quantile(0.8)
data['session_ID'] = 0
session_id = 0
for i in range(0, len(data)):
    if np.isnan(data.iloc[i]['time_stamp_diff']) or data.iloc[i]['time_stamp_diff'] > session_gap_threshold:
        session_id = session_id + 1
        data.iloc[i]['session_ID'] = session_id
    else:
        data.iloc[i]['session_ID'] = session_id
data
'''


# In[9]:


data[data['if_new_session'] == False]['time_stamp_diff'].max()


# In[10]:


session_lengths = data.groupby('session_ID').size()
data = data[np.in1d(data.session_ID, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('item_ID').size()
data = data[np.in1d(data.item_ID, item_supports[item_supports>=12].index)]
session_lengths = data.groupby('session_ID').size()
data = data[np.in1d(data.session_ID, session_lengths[session_lengths>1].index)]
data


# In[11]:


data['user_ID'].nunique()


data = data.sort_values(by = ['session_ID','time_stamp'])
tmax = data['time_stamp'].max()
session_max_times = data.groupby('session_ID')['time_stamp'].max()
session_train = session_max_times[session_max_times < tmax-86400].index
session_test = session_max_times[session_max_times >= tmax-86400].index
train = data[np.in1d(data['session_ID'], session_train)]
test = data[np.in1d(data['session_ID'], session_test)]
test = test[np.in1d(test['item_ID'], train['item_ID'])]
tslength = test.groupby('session_ID').size()
test = test[np.in1d(test['session_ID'], tslength[tslength>1].index)]
print('Training set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(train), train.session_ID.nunique(), train.item_ID.nunique(), train.user_ID.nunique()))
train.to_csv(r'./../data/train.csv', index=False)
print('Testing set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(test), test.session_ID.nunique(), test.item_ID.nunique(), test.user_ID.nunique()))
test.to_csv(r'./../data/test.csv', index=False)




# In[22]:

'''
old_user_train_length = old_user_train.groupby('session_ID').size()
old_user_train = old_user_train[np.in1d(old_user_train.session_ID, old_user_train_length[old_user_train_length>1].index)]
print('Training set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(old_user_train), old_user_train.session_ID.nunique(), old_user_train.item_ID.nunique(), old_user_train.user_ID.nunique()))
old_user_train.to_csv(r'./../data/old_user_train.csv', index=False)

user_test = user_test[np.in1d(user_test.item_ID, old_user_train.item_ID)]
user_test_length = user_test.groupby('session_ID').size()
user_test = user_test[np.in1d(user_test.session_ID, user_test_length[user_test_length>1].index)]
print('Testing set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(user_test), user_test.session_ID.nunique(), user_test.item_ID.nunique(), user_test.user_ID.nunique()))
user_test.to_csv(r'./../data/user_test.csv', index=False)

old_user_test = old_user_test[np.in1d(old_user_test.item_ID, old_user_train.item_ID)]
old_user_test_length = old_user_test.groupby('session_ID').size()
old_user_test = old_user_test[np.in1d(old_user_test.session_ID, old_user_test_length[old_user_test_length>1].index)]
print('Testing set for Old users\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(old_user_test), old_user_test.session_ID.nunique(), old_user_test.item_ID.nunique(), old_user_test.user_ID.nunique()))
old_user_test.to_csv(r'./../data/old_user_test.csv', index=False)

new_user_test = new_user_test[np.in1d(new_user_test.item_ID, old_user_train.item_ID)]
new_user_test_length = new_user_test.groupby('session_ID').size()
new_user_test = new_user_test[np.in1d(new_user_test.session_ID, new_user_test_length[new_user_test_length>1].index)]
print('Testing set for New users\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tUsers: {}'.format(len(new_user_test), new_user_test.session_ID.nunique(), new_user_test.item_ID.nunique(), new_user_test.user_ID.nunique()))
new_user_test.to_csv(r'./../data/new_user_test.csv', index=False)




# In[5]:


import numpy as np
import pandas as pd
import datetime as dt

#PATH_TO_ORIGINAL_DATA = '/data/'
#PATH_TO_PROCESSED_DATA = '/data/'

data = pd.read_csv(r'data/yoochoose-clicks.dat', sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
data.columns = ['SessionId', 'TimeStr', 'ItemId']
data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
del(data['TimeStr'])

session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

tmax = data.Time.max()
session_max_times = data.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-86400].index
session_test = session_max_times[session_max_times >= tmax-86400].index
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]
test = test[np.in1d(test.ItemId, train.ItemId)]
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(r'data/rsc15_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(r'data/rsc15_test.txt', sep='\t', index=False)

tmax = train.Time.max()
session_max_times = train.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-86400].index
session_valid = session_max_times[session_max_times >= tmax-86400].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
train_tr.to_csv(r'data/rsc15_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(r'data/rsc15_train_valid.txt', sep='\t', index=False)
'''

# In[ ]:




