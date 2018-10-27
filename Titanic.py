#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, cross_validation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def cleanData(filePath):
    dataF = pd.read_csv(filePath)
    
    print('Dataset length : ', len(dataF))
    
    dataF['Sex'] = dataF['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
    
    dataF = dataF.drop(['Ticket', 'Cabin', 'Fare'], axis=1)

    dataF['FamilySize'] = dataF['SibSp'] + dataF['Parch']
    dataF = dataF.drop(['SibSp', 'Parch'], axis=1)
    
    # Adding mean Age in place of null values in the 'Age' field
    dataF['Age'] = dataF['Age'].fillna(dataF['Age'].mean())
    
    dataF['Age'] = dataF['Age'].astype(int)
    
    # The RMS Titanic started from Southampton (S) to New York 
    # so let's assume that the 2 NaN values are people who got on the Titanic at S
    dataF['Embarked'] = dataF['Embarked'].fillna('S')
    
    embarkedDictionary = {
    'S' : 1,
    'C' : 2,
    'Q' : 3    
    }
    dataF['Embarked'] = dataF['Embarked'].map(embarkedDictionary)
    
    dataF['PclassAndSex'] = dataF['Pclass']*dataF['Sex']
    
    dataF['PclassAndAge'] = dataF['Pclass']*dataF['Age']
    
    dataF['SexAndAge'] = dataF['Sex']*dataF['Age']
    
    dataF = dataF.drop(['Name'], axis=1)
    
    return dataF

# training the model using the train dataset
train = cleanData('E:/Kaggle/train.csv')
train.head()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    train.loc[:, ('Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize', 'PclassAndSex', 'PclassAndAge', 'SexAndAge')] ,train.loc[:, ('Survived')], test_size=0.20, random_state=6)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, predicted)
print('RandomForestClassifier accuracy for training data : ', accuracy)


# applying the model on test data - COMMENT EVERYTHING AFTER THIS AND ONLY UNCOMMENT IT AT THE VERY END
test = cleanData('E:/Kaggle/test.csv')
test_final = np.array(test.loc[:, ('Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize', 'PclassAndSex', 'PclassAndAge', 'SexAndAge')])
predicted_test = model.predict(test_final)

finalDF = pd.DataFrame({
    'PassengerId': np.array(test['PassengerId']), #function cleanData() returns a data frame so we need to first convert it to an array
    'Survived': np.array(predicted_test)
})

# finalDF -> to display it directly in Jupyter notebook

# to store this in a .csv file
# If it is run multiple times with same .csv file name, then gives 'PermissionError' and doesn't update that file.
# So, only uncomment it at the end

finalDF.to_csv('E:/Kaggle/gender_submission.csv', sep=',') # sep -> to specify delimiter


# In[ ]:




