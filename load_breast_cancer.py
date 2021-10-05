#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


breast_cancer = load_breast_cancer()
print(dir(breast_cancer))


# In[5]:


breast_cancer_data = breast_cancer.data

print(breast_cancer_data.shape)


# In[6]:


breast_cancer_data[0]


# In[7]:


breast_cancer_label = breast_cancer.target
print(breast_cancer_label.shape)
breast_cancer_label


# In[8]:


breast_cancer.target_names


# In[9]:


print(breast_cancer.DESCR)


# In[10]:


breast_cancer.feature_names


# In[11]:


import pandas as pd

print(pd.__version__)


# In[12]:


breast_cancer_df = pd.DataFrame(data=breast_cancer_data, columns=breast_cancer.feature_names)
breast_cancer_df


# In[13]:


breast_cancer_df["label"] = breast_cancer.target
breast_cancer_df


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))


# In[15]:


X_train.shape, y_train.shape


# In[16]:


X_test.shape, y_test.shape


# In[17]:


y_train, y_test


# In[18]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)
print(decision_tree._estimator_type)


# In[19]:


decision_tree.fit(X_train, y_train)


# In[20]:


y_pred = decision_tree.predict(X_test)
y_pred


# In[21]:


y_test


# In[22]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[23]:


# (1) 필요한 모듈 import
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# (2) 데이터 준비
breast_cancer = load_breast_cancer()
breast_cancer_data = breast_cancer.data
breast_cancer_label = breast_cancer.target

# (3) train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=7)

# (4) 모델 학습 및 예측
decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))


# In[24]:


from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, 
                                                    breast_cancer_label, 
                                                    test_size=0.2, 
                                                    random_state=21)

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# In[26]:


from sklearn import svm
svm_model = svm.SVC()

print(svm_model._estimator_type) 


# In[27]:


svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[28]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

print(sgd_model._estimator_type)


# In[29]:


sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[30]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

print(logistic_model._estimator_type)


# In[32]:


from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
breast_cancer.keys()


# In[ ]:


#RandomForestClassifier 모델이 가장 높은 성능을 보임

