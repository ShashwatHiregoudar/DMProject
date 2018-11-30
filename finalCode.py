
##Exploring Traveler data
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('pylab', 'inline')
train_file="deaths-in-india-satp-dfe_Final.csv"

data = pd.read_csv(train_file, header = 0,index_col=None)


# ## Splitting Data into train data and test data

# In[2]:


df_train = data.head(22000)

df_test = data.tail(5000)
df_test.drop('verb',axis = 1, inplace = True)


# ## Conacatenate train and test data for Preprocessing

# In[3]:


data = pd.concat((df_train, df_test), axis=0, ignore_index=True)


# ## Preprocessing 

# In[4]:


#dropping _goden as this attribute contains same value for all of its records
data.drop('_golden', axis = 1, inplace = True)
print("Dropping _golden")

#dropping _unit_state as this attribute contains same value for all of its records
data.drop('_unit_state',axis = 1, inplace = True)
print("Dropping _unit_state")

#dropping _trusted_judgments as this attribute contains same value for all of its records
data.drop('_trusted_judgments',axis = 1, inplace = True)
print("Dropping _trusted_judgments")

#dropping _last_judgment_at as this does not help for our prediction
data.drop('_last_judgment_at',axis = 1, inplace = True)
print("Dropping _last_judgment_at")

#dropping _accuracy as this attribute contains same value for all of its records
data.drop('accuracy',axis = 1, inplace = True)
print("Dropping accuracy")

#dropping canddist as this attribute does not help for our prediction
data.drop('canddist',axis = 1, inplace = True)
print("Dropping canddist")

#dropping civilians_gold as this attribute contains same value for all of its records
data.drop('civilians_gold',axis = 1, inplace = True)
print("Dropping civilians_gold")

#dropping date as this attribute does not help for our prediction
data.drop('date',axis = 1, inplace = True)
print("Dropping date")

#dropping militants_terrorists_insurgents_gold as this attribute  does not contain any value for all of its records
data.drop('militants_terrorists_insurgents_gold',axis = 1, inplace = True)
print("Dropping militants_terrorists_insurgents_gold")

#droppin object as it is a redundant attribute
data.drop('object',axis = 1, inplace = True)
print("Dropping object")

#dropping objectcleanpp as this attribute does nothelp for our prediction
data.drop('objectcleanpp',axis = 1, inplace = True)
print("Dropping objectcleanpp")

#droppin object as it is a redundant attribute
data.drop('objectcount',axis = 1, inplace = True)
print("Dropping objectcount")

#droppin object as does not help for our prediction
data.drop('pid',axis = 1, inplace = True)
print("Dropping pid")

#droppin object as does not help for our prediction
data.drop('rid',axis = 1, inplace = True)
print("Dropping rid")

#dropping security_forces_gold as this attribute does not contain any value for all of its records
data.drop('security_forces_gold',axis = 1, inplace = True)
print("Dropping security_forces_gold")


#dropping objectcleanpp as this attribute does nothelp for our prediction
data.drop('sentence',axis = 1, inplace = True)
print("Dropping sentence")

#dropping sid as this attribute is not helpful possible prediction of future attacks
data.drop('sid',axis = 1, inplace = True)
print("Dropping sid")

#dropping srid as this attribute does not contain any value for all of its records
data.drop('srid',axis = 1, inplace = True)
print("Dropping srid")

#droppin state as it is a redundant attribute
data.drop('state',axis = 1, inplace = True)
print("Dropping state")

#droppin state as it is not helpful possible prediction of future attacks
data.drop('subject',axis = 1, inplace = True)
print("Dropping subject")

#dropping svmlabel2 as this attribute is not helpful possible prediction of future attacks
data.drop('svmlabel2',axis = 1, inplace = True)
print("Dropping svmlabel2")

#dropping svmlabel2prob as this attribute is not helpful possible prediction of future attacks
data.drop('svmlabel2prob',axis = 1, inplace = True)
print("Dropping svmlabel2prob")

#dropping svmobjecttypelab as this attribute contains same value for all of its records
data.drop('svmobjecttypelab',axis = 1, inplace = True)
print("Dropping svmobjecttypelab")

#dropping svmobjecttypeprob as this attribute is not helpful possible prediction of future attacks
data.drop('svmobjecttypeprob',axis = 1, inplace = True)
print("Dropping svmobjecttypeprob")

#dropping total_number_of_people_gold as this attribute does not contain any value for all of its records
data.drop('total_number_of_people_gold',axis = 1, inplace = True)
print("Dropping total_number_of_people_gold")


# ### Computing NULL values in all attributes

# In[5]:


# Compute Null percentage of each feature.
df_all_null = (data.isnull().sum() / data.shape[0]) * 100
#df_all_null = (df_all.isnull().sum())
df_all_null[df_all_null > 0]


# In[6]:


df_all_null.plot(kind="bar",color="red",rot=80)


# ## Filling missing values

# In[7]:


data.to_csv("preprocessed.csv")


# In[8]:


# Fill state_full column
print("Filling state_full column...")
data['state_full'].fillna('NA', inplace=False)
print("Filling state_full column...completed")



data = data.dropna(axis = 0, how ='any') 


# # Computing null after dropping

# In[9]:


# Compute Null percentage of each feature.
df_all_null = (data.isnull().sum() / data.shape[0]) * 100
#df_all_null = (df_all.isnull().sum())
df_all_null[df_all_null > 0]
df_all_null.plot(kind="bar",color="red",rot=80)


# ## Data Transformation

# In[10]:


data['report_date'] = pd.to_datetime(data.report_date)
#converting into Standard date format Y-M-D
data['report_date'] = pd.to_datetime(data['report_date'], format='%Y-%m-%d')


# In[11]:


data['report_date']


# ## Data addition

# In[12]:


#adding new columns
data['attack_month'] = data['report_date'].dt.month
data['attack_day'] = data['report_date'].dt.day


# In[13]:


data.drop('report_date',axis = 1, inplace = True)


# ## Data transformation

# In[14]:


#replacing similar data of the attribute
replaceVerb=data['verb']
import re

for var in replaceVerb:
    if(re.search("killing",str(var)) or re.search("kill",str(var))):
        var2='killed'
        data['verb'].replace(var,var2,inplace=True)
    elif(re.search("arresting",str(var)) or re.search("arrest",str(var))):
        var2='arrested'
        data['verb'].replace(var,var2,inplace=True)
    elif(re.search("surrender",str(var)) or re.search("surrendering",str(var))):
        var2='surrendered'
        data['verb'].replace(var,var2,inplace=True)
    elif(re.search("injuring",str(var)) or re.search("injure",str(var))):
        var2='injured'
        data['verb'].replace(var,var2,inplace=True)


# ## One Hot Encoding 

# In[15]:


def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:4] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1

    return df

# One Hot Encoding
print("One Hot Encoding categorical data...")
columns_to_convert = ['state_full','svmlabel1','districtmatch']

for column in columns_to_convert:
    data = convert_to_binary(data,column)
    data.drop(column, axis=1, inplace=True)
print("One Hot Encoding categorical data...completed")


# In[16]:


#to find total number of deaths based on profession 
security=data['svml_security'].sum()

terrorist=data['svml_terrorist'].sum()

civilian=data['svml_civilian'].sum()

public=data['svml_public'].sum()

private=data['svml_private'].sum()


# In[17]:


# Pie chart
labels = 'security','public', 'terrorist', 'civilian' ,'private'
sizes = [security,public,terrorist,civilian,private]
explode = (0, 0, 0.2, 0.1, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode =explode ,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()


# In[18]:


#from above pie cart we can say that major part of killed were terrorists


# ## Save Preprocessed Data to csv file

# In[19]:


file4 = 'ppppreprocessed.csv'
data.to_csv(file4)


# In[20]:


#create data frame for classification

df_all =  pd.DataFrame(data=data)


# ## Split preprocessed data into train data and test data

# In[21]:


df_train = df_all.head(22000)
df_test = df_all.tail(5000)
df_all1 = df_all


# In[22]:


#dropping column which we want to predivt from test data
df_test.drop('verb',axis = 1, inplace = True)


# In[23]:


df_all.drop('verb',axis = 1, inplace = True)


# In[24]:


#temporary variables
df_train1 = df_train
df_test1 = df_test
df_all1 = df_all


# In[25]:


df_train1.columns


# In[26]:


df_train['verb'].head()


# In[27]:


df_test1.columns


# In[28]:


df_all.columns


# In[29]:


#seeting _unit_id as a index of dataframe
df_train1.set_index('_unit_id', inplace=True)


# ## Label Encoding for Categorical data

# In[30]:


from sklearn.preprocessing import LabelEncoder

id_train = df_train1.index.values
labels = df_train1['verb']

# Label encoding for the categorical data eg: ...NDF -> 7, US -> 10...
le = LabelEncoder()
y = le.fit_transform(labels)
X = df_train1.drop('verb', axis=1, inplace=False)


# In[31]:


X.shape


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
## Spliting of training dataset into 70% training data and 30% testing data randomly
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Gradient Boosting

# In[69]:

'''
## Decision Tree 
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ## Decission tree Classifier

# In[65]:


## Decision Tree 
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ## Gaussian Naive Bayes Classifier

# In[66]:


## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ## SVM Classifier

# In[67]:


## SVM 
from sklearn import svm
clf = svm.SVC(kernel="rbf") 
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # RandomForest classifier

# In[68]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Logistic Regression

# In[70]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # K Nearest Neighbours

# In[72]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Bagging Classifier

# In[71]:


from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Extra Trees Classifier

# In[73]:


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
## Computing accuracy
accuracy = accuracy_score(labels_test , prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''

# # Approach 2

# ## All At once

# In[ ]:


from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.dummy import *
from sklearn.tree import *
from sklearn.gaussian_process import *

ensembles = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    IsolationForest(),
    RandomForestClassifier()
]

neighbors = [
    KNeighborsClassifier(),
    RadiusNeighborsClassifier(),
    NearestCentroid()
]

svms = [
    LinearSVC(),
    NuSVC(),
    SVC(),
]

trees = [
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
]

extraa = [
    DummyClassifier(),
    GaussianProcessClassifier()
]

models = []
models.extend(ensembles)
models.extend(neighbors)
models.extend(svms)
models.extend(trees)
models.extend(extraa)
for model in models:
    try:
        print("---------------------------------------------------------------")
        print(str(model))
        print("---------------------------------------------------------------")
        file = open("Model_output_logger.txt", "a")
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        ## Computing accuracy
        accuracy = accuracy_score(labels_test , prediction)
        print("Accuracy: %.2f%%\n\n\n\n\n\n" % (accuracy * 100.0))
        file.write("model : " + str(model) + "accuracy : " + str(accuracy * 100.0) + "\n\n")
        file.close()
    except:
        print("\n\nerror in")
        print(model)


# In[ ]:




