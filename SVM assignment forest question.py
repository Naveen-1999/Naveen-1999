#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[12]:


import pandas as pd
import numpy as np


# # Import DataSet

# In[13]:


forest_data = pd.read_csv('C:\\Users\\user\\Downloads\\forestfires.csv')
forest_data


# # Data Understanding

# In[14]:


forest_data.dtypes


# In[15]:


forest_data.drop(['month','day'],inplace=True,axis=1)


# In[16]:


forest_data.head()


# In[17]:


forest_data.dtypes


# In[18]:


forest_data['size_category']= forest_data.size_category.map({'small':0,'large':1})


# In[19]:


forest_data.head(15)


# In[20]:


forest_data.size_category.unique()


# In[21]:


forest_data.size_category.value_counts()


# # Data Preparation

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler=scaler.fit_transform(X)
X_scaler


# # Model Building

# In[23]:


X = forest_data.drop(['size_category'],axis=1)
y = forest_data['size_category']


# In[24]:


X


# In[26]:


y


# In[27]:


X.std()


# In[28]:


X_scaler.std()


# In[29]:


X_scaler.mean()


# In[30]:


X.shape,y.shape


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=123)


# In[33]:


X_train.shape,y_train.shape


# In[34]:


X_test.shape,y_test.shape


# In[56]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,roc_auc_score,classification_report


# In[36]:


from sklearn.svm import SVC


# # Kernel - Linear
# 
# ## Model Training without scaling | Model Testing | Model Evaluation

# In[52]:


svc_classifier = SVC(kernel='linear')
svc_classifier.fit(X_train,y_train)
y_pred=svc_classifier.predict(X_test)


# In[38]:


accuracy_score(y_test,y_pred)


# In[39]:


conf_matrix =confusion_matrix(y_test,y_pred)
conf_matrix


# In[40]:


print("Overall Accuracy : ",round(accuracy_score(y_test,y_pred),4))
print("Precision        : ",round(precision_score(y_test,y_pred),4))
print("Recall           : ",round(recall_score(y_test,y_pred),4))
print("AUC Score        : ",round(roc_auc_score(y_test,y_pred),4))


# In[66]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.heatmap(data = conf_matrix, annot=True, linewidths=0.8,cmap='plasma')
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)


# # Kernel - Radial basis function

# In[42]:


svc_classifier_1 = SVC(kernel='rbf')
svc_classifier_1.fit(X_train,y_train)
y_pred_1=svc_classifier_1.predict(X_test)


# In[43]:


accuracy_score(y_test,y_pred_1)


# In[44]:


conf_matrix_1=confusion_matrix(y_test,y_pred_1)
conf_matrix_1


# In[45]:


print("Overall Accuracy : ",round(accuracy_score(y_test,y_pred_1),4))
print("Precision        : ",round(precision_score(y_test,y_pred_1),4))
print("Recall           : ",round(recall_score(y_test,y_pred_1),4))
print("AUC Score        : ",round(roc_auc_score(y_test,y_pred_1),4))


# In[67]:


sns.heatmap(data = conf_matrix_1,annot=True,linewidth=0.8,cmap='plasma')
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)


# # Kernel - 'Polynomial'

# In[47]:


svc_classifier_2 = SVC(kernel='poly')
svc_classifier_2.fit(X_train,y_train)
y_pred_2=svc_classifier_2.predict(X_test)


# In[48]:


y_pred_2


# In[49]:


conf_matrix_2=confusion_matrix(y_test,y_pred_2)
conf_matrix_2


# In[50]:


print("Overall Accuracy : ",round(accuracy_score(y_test,y_pred_2),4))
print("Precision        : ",round(precision_score(y_test,y_pred_2),4))
print("Recall           : ",round(recall_score(y_test,y_pred_2),4))
print("AUC Score        : ",round(roc_auc_score(y_test,y_pred_2),4))


# In[69]:


sns.heatmap(data=conf_matrix_2,annot = True ,linewidth=0.7 , cmap='plasma')
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)
plt.show()


# # Kernel - 'Sigmoid'

# In[53]:


svc_classifier_3 = SVC(kernel='sigmoid')
svc_classifier_3.fit(X_train,y_train)
y_pred_3=svc_classifier_3.predict(X_test)


# In[54]:


conf_matrix_3=confusion_matrix(y_test,y_pred_3)
conf_matrix_3


# In[55]:


print("Overall Accuracy : ",round(accuracy_score(y_test,y_pred_3),4))
print("Precision        : ",round(precision_score(y_test,y_pred_3),4))
print("Recall           : ",round(recall_score(y_test,y_pred_3),4))
print("AUC Score        : ",round(roc_auc_score(y_test,y_pred_3),4))


# In[70]:


sns.heatmap(data=conf_matrix_3,annot = True ,linewidth=0.7 , cmap='plasma')
plt.xlabel("Prediction")
plt.ylabel("Actual labels")
plt.title("Prediction Vs Actual",size = 20)
plt.show()


# # Classifier 1 if c=1,gamma=0.001,Kernal=rbf
# 
# 
# As we are classifying the values by using the two hyperparameters for two non linearly seperable data points 
# 
# 1)C
# 
# 2)Gamma

# In[57]:


def values(gamma,c=1):
    model= SVC( C=c,kernel='rbf', degree=3,gamma=gamma)
    model.fit(X_train,y_train)
    pred_model=model.predict(X_train)
    print(accuracy_score(y_train,pred_model))
    print(confusion_matrix(y_train,pred_model))
    print(classification_report(y_train,pred_model))
    return model


# In[58]:


values(0.001)


# # classifier 2 - c=1,gamma=0.01

# In[59]:


values(0.01)


# # Classifier 3- c=1,gamma=0.1

# In[60]:


values(0.1)


# **If the gamma value is low ,it indicates the larger similarity distance which results in more points being grouped together.so that the it doesnt capture the shape well.**

# **If the gamma value is high(i.e is from 0.001-0.01-0.1-1),we can say that it can give better accuracy and shape well.**

# **so we kept the c value constant as 1 so that it can reduce the missclassification points with small marginal boundary.**

# In[75]:


def values(gamma,c=1):
    model1= SVC( C=c,kernel='poly', degree=3,gamma=gamma)
    model1.fit(X_train,y_train)
    pred_model_1=model1.predict(X_train)
    print(accuracy_score(y_train,pred_model_1))
    print(confusion_matrix(y_train,pred_model_1))
    print(classification_report(y_train,pred_model_1))
    return model1


# In[76]:


values(0.001)


# In[86]:


def values(gamma,c=0.1):
    model2= SVC( C=c,kernel='rbf', degree=3,gamma=gamma)
    model2.fit(X_train,y_train)
    pred_model_2=model2.predict(X_train)
    print(accuracy_score(y_train,pred_model_2))
    print(confusion_matrix(y_train,pred_model_2))
    print(classification_report(y_train,pred_model_2))
    return model2


# In[87]:


values(0.1)


# **so if c value is small it gives the lower weightage to the missclassified points and increasing the marginal boundary**
