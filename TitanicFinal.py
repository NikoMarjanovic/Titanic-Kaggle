# Importing the libraries

# libraries for data manipulation
import pandas as pd
import numpy as np

# libraries for plotting and visualisation
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() # setting seaborn default for plots

from collections import Counter

# libraries for classifiers and ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
import xgboost as xgb

#for label encoding
from sklearn.preprocessing import LabelEncoder

# libraries for tuning hyperparameters and score evaluation
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, learning_curve

# Importing the dataset
trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')
# Union of datasets for feature engineering 
dataset = pd.concat([trainset, testset]).reset_index(drop=True)

#sample first rows of data
dataset.head()

g = sns.heatmap(trainset[
    ["Survived","Pclass", "SibSp","Parch","Age","Fare"]].corr(),
                annot=True, fmt = ".2f", cmap = "coolwarm")

# Explore SibSp feature vs Survived
f = sns.catplot(x="SibSp",y="Survived",data=trainset,kind="bar", height= 4,
palette = "muted")
plt.title("SibSp Survival probability",fontsize=18)
f.despine(left=True)
f = f.set_ylabels("Survival probability")

# Explore Parch feature vs Survived
g  = sns.catplot(x="Parch",y="Survived",data=trainset,kind="bar",height = 4 , 
palette = "muted")
plt.title("Parch Survival probability",fontsize=18)
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Age distibution 
g = sns.kdeplot(trainset["Age"][(trainset["Survived"] == 0) & (trainset["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(trainset["Age"][(trainset["Survived"] == 1) & (trainset["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# Explore Parch feature vs Survived
g  = sns.catplot(x="Sex",y="Survived",data=trainset,kind="bar",height = 4 , 
palette = "muted")
plt.title("Sex Survival probability",fontsize=18)
g.despine(left=True)
g = g.set_ylabels("survival probability")

#Count of null values per category
ax = dataset.isnull().sum().plot(kind='bar')
ax.set_title("Count of null values per category", fontsize=18)
#ax.set_ylabel("# null values", fontsize=10);
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1, p.get_height() * 1.015))


trainset["Cabin"]=trainset["Cabin"].astype(str).str[0]
print(trainset['Cabin'].value_counts())
# Explore Cabin-NoCabin vs Survived
g  = sns.catplot(x="Cabin" ,y="Survived",data=trainset,kind="bar",height = 6 , aspect= 2, 
palette = "muted")
plt.title("Cabin Survival probability",fontsize=18)
g.despine(left=True)
g = g.set_ylabels("survival probability")


#Split title from name
dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# a litle info
print(dataset['Title'].value_counts())


dataset.groupby('Title')['Survived'].value_counts() / dataset.groupby('Title')['Survived'].count()



##Thank you 
##https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
## for this next part

#replacing all titles with mr, mrs, miss, master with the exception of rev, because they have a high chance of dying.

def replace_titles(x):
    title=x['Title']
    if title in ['Sir', 'Don', 'Major', 'Capt', 'Jonkheer', 'Col']:
        return 'Mr'
    elif title in ['Lady', 'Dona',  'the Countess', 'Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
dataset['Title']=dataset.apply(replace_titles, axis=1)
print(dataset['Title'].value_counts())



g = sns.catplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Mr", "Miss", "Mrs", "Master", "Rev"])
g = g.set_ylabels("survival probability")


# Index of Nan age rows
idNan = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in idNan:
    age_med = dataset["Age"][(dataset['Title'] == dataset.iloc[i]["Title"])].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        ## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
        dataset['Age'].iloc[i] = age_pred
    else :
        ## complete missing age with median by title group
        dataset['Age'].iloc[i] = age_med
        
#Nan cleaning verification
print('Train columns with null values:\n', 
dataset.isnull().sum())
print("-"*10)

#complete embarked with mode(highest frequency value)
dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
dataset['Cabin'].fillna("n", inplace = True)
dataset["Cabin"]=dataset["Cabin"].astype(str).str[0]
#Nan cleaning verification
print('columns with null values:\n', dataset.isnull().sum())
print("-"*10)

dataset.head()


##Next in list is the ticket feature. 
print(dataset['Ticket'].value_counts())
## 681 ditinct values seems like a lot
dataset.loc[dataset['Ticket']=="CA. 2343"]
## but the data for each groups of tickets seems pretty correlated to 
## the survival outcome.

##Let's explore this feature further

trainset['Ticket'].isin(testset['Ticket']).value_counts()
## we have 207 matching ticket values in the test and training set, this might help us in the accuracy of our results.


label = LabelEncoder()
dataset['Ticket'] = label.fit_transform(dataset['Ticket'])

## lets see if all this trouble might be useful
g = sns.heatmap(dataset.iloc[0:891,:][
    ["Survived","Pclass", "SibSp","Parch","Age","Fare", "Ticket"]].corr(),
                annot=True, fmt = ".2f", cmap = "coolwarm")
##we have a correlation of 0.16 which is not as close to the 0.2 limit we would usually use but 
## we can keep it for now to see if it might change later.




#Count the family size with siblings, s.o., parent or child.(+1 to include passenger himself)
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
#convert to boolean if the passenger is alone or not.
# Create new feature of family size
dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

dataset.drop(['Name', 'FamilySize'], axis = 1, inplace = True)

label = LabelEncoder()
dataset['Sex'] = label.fit_transform(dataset['Sex'])
dataset['Embarked'] = label.fit_transform(dataset['Embarked'])
dataset['Title'] = label.fit_transform(dataset['Title'])
dataset['Cabin'] = label.fit_transform(dataset['Cabin'])
dataset.iloc[0:891,:].corr(method='spearman')



todummy = ['Pclass', 'Sex', 'Embarked', 'Title', "Cabin"]
for feature in todummy:  
    dummies = pd.get_dummies(dataset[feature], prefix = feature, drop_first= True)
    dataset = pd.concat([dataset, dummies], axis = 1)
    dataset.drop([feature], axis=1, inplace = True)
dataset.head()

dataset.iloc[890:893,:]

X = dataset.iloc[0:891, 2: ].values
y = dataset.iloc[0:891, 1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

kfold = StratifiedKFold(n_splits=10)
random_state = 0
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(SVC(kernel='linear'))
classifiers.append(GaussianNB())
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(xgb.XGBClassifier())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(
        classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))


cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({
    "CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":[
    "SVC","Linear SVM","Naive Baye's","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting",
    "MultipleLayerPerceptron","KNeighboors","LogisticRegression",
    "LinearDiscriminantAnalysis","XGB"]})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, refit = True, verbose = 1)
gsSVMC.fit(X_train,y_train)
SVMC_best = gsSVMC.best_estimator_
# Best score
print(gsSVMC.best_score_)
print(gsSVMC.best_params_)

RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, refit = True, verbose = 1)
gsRFC.fit(X_train,y_train)
RFC_best = gsRFC.best_estimator_
# Best score
print(gsRFC.best_score_)
print(gsRFC.best_params_)


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, refit = True, verbose = 1)

gsGBC.fit(X_train,y_train)

GBC_best = gsGBC.best_estimator_

# Best score
print(gsGBC.best_score_)
print(gsGBC.best_params_)


#XGB

XGB = xgb.XGBClassifier()
gb_param_grid =     {"n_estimators": [125, 150],
     "max_depth": [6, 8, 10],
     "learning_rate": [0.01, 0.1, 0.001]}

gsXGB= GridSearchCV(XGB,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, refit = True, verbose = 1)

gsXGB.fit(X_train,y_train)

XGB_best = gsXGB.best_estimator_

# Best score
print(gsXGB.best_score_)
print(gsXGB.best_params_)


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, refit =True, verbose = 1)

gsExtC.fit(X_train,y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
print(gsExtC.best_score_)
print(gsExtC.best_params_)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RFC Learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsXGB.best_estimator_,"XGB learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,y_train,cv=kfold)


test_Survived_RFC = pd.Series(RFC_best.predict(X_test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(X_test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(X_test), name="SVC")
test_Survived_XGB = pd.Series(XGB_best.predict(X_test), name="XGB")
test_Survived_GBC = pd.Series(GBC_best.predict(X_test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_XGB,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)

X_train = dataset.iloc[:891, 2: ].values
y_train = dataset.iloc[:891, 1].values
X_test = dataset.iloc[891:, 2: ].values
y_test = dataset.iloc[891:, 1].values

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('xgb',XGB_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)
votingC.predict(X_test)


y_pred = votingC.predict(X_test).astype(int)
submission = pd.DataFrame({
        "PassengerId": dataset.iloc[891:, 0].values,
        "Survived": y_pred
    })

submission.to_csv("submission.csv",index=False)