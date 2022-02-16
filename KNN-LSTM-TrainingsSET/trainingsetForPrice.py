#!/usr/bin/env python
# coding: utf-8

# In[3]:




# In[6]:


import pandas as pd
import numpy as np

df = pd.read_csv('tool_rentals.csv', usecols=[1])


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)



# In[7]:


# change epoch number for creating column amount as epoch number , its a shifting process
epoch_number=10
epoch_columns= []
for i in range(epoch_number+1):
    epoch_columns.append(str(i+1))
print(epoch_columns)


# In[10]:




epoch_columns= ()

newdf=pd.DataFrame(columns = epoch_columns)

k=0
for i in range(len(df)-(epoch_number)):
    simple_list = np.zeros(shape=(0,epoch_number+1))
    for j in range(k,(epoch_number+k+1)):
        simple_list = np.append(simple_list,df.iloc[j].values[0])
    listdf= pd.DataFrame(data=(simple_list))
    
    newdf = pd.concat([newdf,listdf.T] )
    k=k+1
        



# In[11]:


newdf.dropna(inplace=True)
newdf.reset_index(drop=True, inplace=True)



# In[6]:


# change epoch number in file name here for renaming process
newdf.to_csv('epoch10_toolset12.csv',index=False)


# In[ ]:




