#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[5]:


digit = load_digits()
print(dir(digit))


# In[6]:


digit.keys()


# In[7]:


digit_data = digit.data

print(digit_data.shape)


# In[8]:


digit_data[0]


# In[13]:


digit_label = digit.target
print(digit_label.shape)
digit_label


# In[14]:


digit.target_names


# In[15]:


print(digit.DESCR)


# In[18]:


digit.feature_names


# In[25]:


import pandas as pd

print(pd.__version__)


# In[47]:


digit_df = pd.DataFrame(data=digit_data, columns=digit.feature_names)
digit_df


# In[48]:


digit_df["label"] = digit.target
digit_df


# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digit_data, 
                                                    digit_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))


# In[29]:


X_train.shape, y_train.shape


# In[30]:


X_test.shape, y_test.shape


# In[31]:


y_train, y_test


# In[32]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[33]:


decision_tree.fit(X_train, y_train)


# In[34]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[35]:


y_test


# In[36]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[39]:


# (1) 필요한 모듈 import
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비
digit = load_digits()
digit_data = digit.data
digit_label = digit.target

# (3) train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digit_data, 
                                                    digit_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))


# In[40]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(digit_data, 
                                                    digit_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[41]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type)


# In[42]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[43]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[44]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[45]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[46]:


from sklearn.datasets import load_digits

digits = load_digits()
digits.keys()


# In[ ]:


#svm 모델이 가장 높은 성능을 보임

