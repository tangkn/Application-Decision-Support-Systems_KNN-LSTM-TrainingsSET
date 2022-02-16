#!/usr/bin/env python
# coding: utf-8

# In[15]:




# In[16]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn import svm, preprocessing 
import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os
import plotly.graph_objs as go


# In[ ]:
df=pd.read_csv('epoch5_toolset12.csv')


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
df=df_normalized
df


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
X= df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error


# In[ ]:


X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.3)


# In[ ]:


#Taking odd integers as K vales so that majority rule can be applied easily. 
neighbors = np.arange(1, 20, 2)
scores = []


# In[ ]:


for k in neighbors:   # running for different K values to know which yields the max accuracy. 
    clf = KNeighborsRegressor(n_neighbors = k,  weights = 'distance', p=1)
    clf.fit(X_tr, y_tr)
    score = cross_val_score(clf, X_tr, y_tr, cv = 10)
    scores.append(score.mean())


# In[17]:


mse = [1-x for x in scores]

# In[18]:


trace0 = go.Scatter(
    y = mse,
    x = neighbors,
    mode = 'lines+markers', 
    marker = dict(
        color = 'rgb(150, 10, 10)'
    )
)
layout = go.Layout(
    title = '', 
    xaxis = dict(
        title = 'K value', 
        tickmode = 'linear'
    ),
    yaxis = dict(
        title = 'CV Error',
    )
)
fig = go.Figure(data = [trace0], layout = layout)
iplot(fig, filename='basic-line')
plot(fig)

# In[20]:


# Training the model on Optimal K.


optimal_k = neighbors[mse.index(min(mse))]
print("Optimal K: ", optimal_k)
clf_optimal = KNeighborsRegressor(n_neighbors = optimal_k)
clf_optimal.fit(X_tr, y_tr)
y_pred = clf_optimal.predict(X_test)
acc = clf_optimal.score(X_test, y_test)
print("Accuracy: ", acc*100)
print("RMS Error: ", mean_squared_error(y_test, y_pred))


# In[21]:




# In[ ]:





# In[ ]:





# In[ ]:




