# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd  
import seaborn as sns

# %% [markdown]
# ## loading data

# %%
titanic_data = pd.read_csv("titanic dataset.csv")
titanic_data.head()

# %% [markdown]
# ## Analysing Data

# %%
sns.countplot(x = 'Survived',data=titanic_data)


# %%
sns.countplot(x = "Survived",hue='Sex',data=titanic_data)


# %%
sns.countplot(x = "Survived",hue='Pclass',data=titanic_data)


# %%
titanic_data['Age'].plot.hist()


# %%
sns.countplot(x = "Survived",hue='Age',data=titanic_data)


# %%
titanic_data['Fare'].plot.hist(bins = 20 ,figsize=(10,5))


# %%
titanic_data.info()

# %% [markdown]
#  ## Data Wrangling

# %%
titanic_data.isnull()


# %%
titanic_data.isnull().sum()


# %%
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')


# %%
titanic_data.head()


# %%
titanic_data.drop('Cabin',axis=1,inplace=True)
titanic_data.head()


# %%
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)


# %%
titanic_data.isnull().sum()

# %% [markdown]
# ## Removing stings and conert it into dummies variables

# %%
gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
gender.head()


# %%
embarked = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embarked.head()


# %%
Pclass = pd.get_dummies(titanic_data['Pclass'],drop_first=True)
Pclass.head()


# %%
titanic_data = pd.concat([titanic_data,gender,embarked,Pclass],axis=1)
titanic_data.head()


# %%

titanic_data.drop(['Sex','PassengerId','Name','Ticket','Embarked','Pclass'],axis=1,inplace= True)


# %%
titanic_data.head()

# %% [markdown]
# ### Tarin and Test Data

# %%
x = titanic_data.drop('Survived',axis=True)
y= titanic_data['Survived']


# %%
from sklearn.model_selection import train_test_split


# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# %%
from sklearn.linear_model import LogisticRegression


# %%
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# %%
predictions = logmodel.predict(X_test)


# %%
from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)*100


