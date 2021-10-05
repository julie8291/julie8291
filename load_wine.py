#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


wine = load_wine()
print(dir(wine))


# In[4]:


wine_data = wine.data

print(wine_data.shape)


# In[5]:


wine_data[0]


# In[6]:


wine_label = wine.target
print(wine_label.shape)
wine_label


# In[7]:


wine.target_names


# In[8]:


print(wine.DESCR)


# In[9]:


wine.feature_names


# In[10]:


import pandas as pd

print(pd.__version__)


# In[11]:


wine_df = pd.DataFrame(data=wine_data, columns=wine.feature_names)
wine_df


# In[12]:


wine_df["label"] = wine.target
wine_df


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))


# In[14]:


X_train.shape, y_train.shape


# In[15]:


X_test.shape, y_test.shape


# In[16]:


y_train, y_test


# In[17]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[18]:


decision_tree.fit(X_train, y_train)


# In[19]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[20]:


y_test


# In[21]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[22]:


# (1) 필요한 모듈 import
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비
wine = load_wine()
wine_data = wine.data
wine_label = wine.target

# (3) train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))


# In[23]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(wine_data, 
                                                    wine_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[24]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[25]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[26]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[27]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[29]:


from sklearn.datasets import load_wine

wine = load_wine()
wine.keys()


# In[30]:


#RandomForestClassifier 모델이 가장 높은 성능을 보임


# In[31]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

