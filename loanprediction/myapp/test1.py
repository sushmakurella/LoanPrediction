import numpy as np
import pandas as pd
# For visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Set Seaborn styles
sb.set()

# For implementing pipeline
from sklearn.pipeline import Pipeline

# For Scaling the data
from sklearn.preprocessing import StandardScaler

# For Classification
from sklearn.neighbors import KNeighborsClassifier

# For Splitting the data for training and Validation
from sklearn.model_selection import train_test_split

# For creating model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Principal Component Analysis for Dimensionality Reduction
from sklearn.decomposition import PCA

# For Shuffling the DataFrame
from sklearn.utils import shuffle
# TODO : Load previous individuals loan data
loan_train = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/train_u6lujuX_CVtuZ9i.csv')
print("The Loan Prediction Dataset has")
print("\t\tNumber of Factors : \t", loan_train.shape[1] - 1)
print("\t\tNumber of Samples : \t", loan_train.shape[0])

print(loan_train.describe())

print('Loan Data Factors : \n')
j = 1
for i in loan_train.columns:
    print(j, '.', i)
    j += 1

# Let's check is there any null values on Loan_Status
print(loan_train['Loan_Status'].isnull().sum())

#DO : To get unique values and value_counts on Loan_Status
print("Unique values : ", loan_train['Loan_Status'].unique())
print("Unique values counts : \n", loan_train['Loan_Status'].value_counts())

# Let's plot the Unique value counts
# plt.figure(figsize=(6, 6))
# sb.countplot(x = 'Loan_Status', data = loan_train)

loan_train.isnull().sum()

print(loan_train.shape)

# 1. Loan ID
loan_train['Loan_ID'].describe()

# TODO : Describe Gender column
loan_train['Gender'].describe()

print("Number of null values : ", loan_train['Gender'].isnull().sum())
print("Unique values : ", loan_train['Gender'].unique())
print("Value counts : \n", loan_train['Gender'].value_counts())

# plt.figure(figsize=(6, 6))
# sb.countplot(x = 'Gender', data = loan_train)
# plt.savefig('../images/gender_counts_0.png')

# TODO : Print the samples having null values in gender column
samples_with_null_values_on_gender_column = loan_train[loan_train['Gender'].isnull()]
print(samples_with_null_values_on_gender_column)

# Initialize Gender predictor columns list
GENDER_PREDICTOR_COLUMNS = ['Dependents', 'ApplicantIncome', 'LoanAmount', 'Property_Area', 'Gender']

# Let's get the rows not having null values on gender column
samples_without_null_values_on_gender_column = loan_train[~loan_train['Gender'].isnull()][GENDER_PREDICTOR_COLUMNS]
print("Number of Samples Before dropping samples having null values in predictor columns for Gender column: ", 
     samples_without_null_values_on_gender_column.shape[0])

# Dropping samples having null values in predictor columns
samples_without_null_values_on_gender_column = samples_without_null_values_on_gender_column.dropna(how = 'any')
print("Number of Samples After dropping samples having null values in predictor columns for Gender column: ", 
     samples_without_null_values_on_gender_column.shape[0])

# Conveting categorical values on Dependents column to numerical values
samples_without_null_values_on_gender_column['Dependents'] = samples_without_null_values_on_gender_column['Dependents'].apply(
    lambda x : {'0': 0, '1':1, '2':2, '3+':3}[x]
)
# Conveting categorical values on Property_Area column to numerical values
samples_without_null_values_on_gender_column['Property_Area'] = samples_without_null_values_on_gender_column['Property_Area'].apply(
    lambda x : {'Urban': 2, 'Semiurban':1.5, 'Rural':1}[x]
)

# Building Gender Predictor using Pipeline and KNeigborsClassifiers
Gender_predictor = Pipeline(steps = [
    ('scaler', StandardScaler()),
    ('gen_predictor', KNeighborsClassifier(n_neighbors = 1))
])
Gender_predictor.fit(samples_without_null_values_on_gender_column.iloc[:, :-1].values,
                    samples_without_null_values_on_gender_column.iloc[:, -1].values)
# Function which fills null values on Gender column
def gender_null_value_filler(df, Gender_predictor):
    for row in range(df.shape[0]):
        if df.loc[row, 'Gender'] is np.nan:
            X = loan_train.loc[row, GENDER_PREDICTOR_COLUMNS[:-1]].values.reshape(1, -1)
            X[0][0] = {'0': 0, '1':1, '2':2, '3+':3}[X[0][0]]
            X[0][3] = {'Urban': 2, 'Semiurban':1.5, 'Rural':1}[X[0][3]]
            df.loc[row, 'Gender'] = Gender_predictor.predict(X)
    return df
loan_train = gender_null_value_filler(loan_train, Gender_predictor)
print(loan_train['Gender'].dtype)
loan_train['Gender'] = loan_train['Gender'].astype(str)  # Convert to string type
loan_train['Gender'] = loan_train['Gender'].apply(lambda x: {'Male': 1, 'Female': 0}.get(x, -1))  # Use .get() to handle other values

# TODO : Encoding Gender Column - Male : 1, Female : 0
loan_train['Gender'] = loan_train['Gender'].replace({'Male': 1, 'Female' : 0})
# TODO : Describing Married column
print(loan_train['Married'].describe())
print("Number of null values : ", loan_train['Married'].isnull().sum())
print("Unique values : ", loan_train['Married'].unique())
print("Value counts : \n", loan_train['Married'].value_counts())
# Let's plot the Unique value counts
# plt.figure(figsize = (6, 6))
# sb.countplot(x = 'Married', data = loan_train)
# plt.savefig('../images/married_counts_0.png')
# TODO : Print the samples having null values in gender column
samples_with_null_values_on_gender_column = loan_train[loan_train['Married'].isnull()]
print(samples_with_null_values_on_gender_column)

loan_train[loan_train['Married'] == 'Yes'].describe(include = 'all').iloc[:, :]
# Let's fill null values in Married columns with 'Yes'
loan_train['Married'] = loan_train['Married'].fillna('Yes')
# TODO : To know the unique value counts
loan_train['Married'].value_counts()
# TODO : encoding categorical values into numerical values
loan_train['Married'] = loan_train['Married'].apply(lambda x : {'Yes' : 1, 'No' : 0}[x])
print(# TODO : Describing Dependents column
loan_train['Dependents'].describe())

print("Number of null values : ", loan_train['Dependents'].isnull().sum())
print("Unique values : ", loan_train['Dependents'].unique())
print("Value counts : \n", loan_train['Dependents'].value_counts())
# plt.figure(figsize=(6, 6))
# sb.countplot(x = 'Dependents', data = loan_train)
# plt.savefig('../images/dependents_counts_0.png')
# plt.show()
# TODO : Display the 15 rows having null values in Dependents column
print(loan_train[loan_train['Dependents'].isnull()])
# TODO : Function for filling null values on dependents columns
def dependents_null_value_filler(df):
    for row in range(df.shape[0]):
        if df.loc[row, 'Dependents'] is np.nan:
            df.loc[row, 'Dependents'] = str(df.loc[row, 'Married'])
    return df
# TODO : Fill null values on Dependents column
loan_train = dependents_null_value_filler(loan_train)
# TODO : Encoding Categorical data into NUmerical Data
loan_train['Dependents'] = loan_train['Dependents'].apply(lambda x : {'0' : 0, '1' : 1, '2' : 2, '3+' : 3}[x])
# TODO : Descriptive Statistics on EDucation columns
print(loan_train['Education'].describe())
# TODO : Number of nul values on Education column
print(loan_train['Education'].isnull().sum())
# TODO : Encoding categorical data into Numerical data
loan_train['Education'] = loan_train['Education'].apply(lambda x : {'Graduate' : 1, 'Not Graduate' : 0}[x])
# plt.figure(figsize=(6, 6))
# sb.countplot(x = 'Education', data = loan_train)
# # plt.savefig('../images/education_counts_0.png')
# plt.show()
# TODO : Descriptive Statistics on Self_Employed column
print(loan_train['Self_Employed'].describe())
# TODO : Uniques and Values count on Self_Employed column
print("Number of null values : ", loan_train['Self_Employed'].isnull().sum())
print("Unique values : ", loan_train['Self_Employed'].unique())
print("Value counts : \n", loan_train['Self_Employed'].value_counts())

# TODO : Filling Null values with No on Self_Employed column
loan_train['Self_Employed'].fillna(value = 'No', inplace = True)

# TODO : Encoding Self_Employed as 1 and Not Self_Employed as 0
loan_train['Self_Employed'] = loan_train['Self_Employed'].apply(lambda x : {'Yes' : 1, 'No' : 0}[x])
# Let's get the knowledge about Applicant Income

# TODO : Descriptive Statistics on Applicant Income
print(loan_train['ApplicantIncome'].describe())
# TODO : Check for null values on ApplicantIncome column
print(loan_train['ApplicantIncome'].isnull().sum())
# TODO : Distribution of Applicant Income

# plt.figure(figsize = (14, 6))
# sb.distplot(loan_train['ApplicantIncome'], rug = True, bins = 100, color='r')
# plt.savefig('../images/ApplicantIncomeDistribution.png')

# TODO : Displaying the applicants having income more than 20,000
print(loan_train[loan_train['ApplicantIncome'] > 20000])

# TODO : Let's know the contribution of Applicant Income on determining Loan_Status 
sb.catplot(x = 'Loan_Status', y = 'ApplicantIncome', data = loan_train)

# TODO : Descriptive Statistics on Co-applicant's Income
print(loan_train['CoapplicantIncome'].describe())

# TODO : Check for null values on co-applicant income column
print(loan_train['CoapplicantIncome'].isnull().sum())

# plt.figure(figsize = (14, 6))
# sb.distplot(loan_train['CoapplicantIncome'], rug = True, color = 'r')
# plt.savefig('../images/CoapplicantIncomeDistribution.png')

# TODO : Let's get the different values counts on CoapplicantIncome column
print(loan_train['CoapplicantIncome'].value_counts())

# TODO : Descriptive Statistics on LoanAmount
print(loan_train['LoanAmount'].describe())

# TODO : Distribution of LoanAmount
# plt.figure(figsize = (14, 6))
# sb.distplot(loan_train['LoanAmount'], rug = True, color = 'r')
# plt.savefig('../images/LoanAmountDistribution.png')

# TODO : Let's know the different LoanAmount
print(loan_train['LoanAmount'].value_counts())

# TODO : Count of Null values on LoanAmount column
print(loan_train['LoanAmount'].isnull().sum())
# TODO : Display the Samples having null values on LoanAmount
print(loan_train[loan_train['LoanAmount'].isnull()])

# TODO : To know if yes in LoanStatus, then what is the average LoanAmount 
#        and if no in LoanStatus, then what is the average LoanAmount by using GroupBy in LoanStatus

print(loan_train[~loan_train['LoanAmount'].isnull()].groupby('Loan_Status').describe().T.loc['LoanAmount'])


# TODO : Filling Above values on LoanAmount column based on LoanStatus.
def LoanAmount_null_values_filler(df):
    for row in range(df.shape[0]):
        if pd.isnull(df.loc[row, 'LoanAmount']):
            if df.loc[row, 'Loan_Status'] == 'Y':
                df.loc[row, 'LoanAmount'] = 151.22
            elif df.loc[row, 'Loan_Status'] == 'N':
                df.loc[row, 'LoanAmount'] = 144.29
            else:
                pass
    return df
# TODO : Filling null values on LoanAmount
loan_train = LoanAmount_null_values_filler(loan_train)

# TODO : Descriptive Statistics on Loan_Amount_Term
print(loan_train['Loan_Amount_Term'].describe())

# TODO : Number of null values on Loan_Amount_Term
print(loan_train['Loan_Amount_Term'].isnull().sum())
# TODO : Unique Values count in Loan_Amount_Term column
print(loan_train['Loan_Amount_Term'].value_counts())

# plt.figure(figsize=(6, 6))
# sb.countplot(x = 'Loan_Amount_Term', data = loan_train)
# # plt.savefig('../images/term_counts.png')
# plt.show()

# TODO : Display the applicant samples aving null values on Loan_Amount_Term
print(loan_train[pd.isnull(loan_train['Loan_Amount_Term'])])

# TODO : To know if yes in LoanStatus, then what is the average Loan_Amount_Term
#        and if no in LoanStatus, then what is the average Loan_Amount_Term by using GroupBy in LoanStatus

print(loan_train[~loan_train['Loan_Amount_Term'].isnull()].groupby('Loan_Status').describe().T.loc['Loan_Amount_Term'])

# TODO : Fill null values on Loan_Amount_Term
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna(value = 360)

# TODO : Descriptive Statistics on Credit_History
print(loan_train['Credit_History'].describe())

# TODO : Number of null values on Credit_History
print(loan_train['Credit_History'].isnull().sum())

# TODO : Unique values count on Credit_History
print(loan_train['Credit_History'].value_counts())

print(loan_train[loan_train['Credit_History'].isnull()])


# TODO : To know if yes in LoanStatus, then what is the average Credit_History
#        and if no in LoanStatus, then what is the average Credit_History by using GroupBy in LoanStatus

print(loan_train[~loan_train['Credit_History'].isnull()].groupby('Loan_Status').describe().T.loc['Credit_History'])

# TODO : Filling null values on Credit_History
loan_train['Credit_History'] = loan_train['Credit_History'].fillna(value = 1.0)

# TODO : Descriptive Statistics on Property_Area
print(loan_train['Property_Area'].describe())

# TODO : Number of null values on Property_Area
print(loan_train['Property_Area'].isnull().sum())

# TODO : Unique values count in Property_Area column
print(loan_train['Property_Area'].value_counts())

# TODO : To get Property_Area Dummies
Property_Area_Dummies = pd.get_dummies(loan_train['Property_Area'])
print(Property_Area_Dummies)

# TODO : Create Separate column for Rural and Urban Property_Area
loan_train['Property_Area_Rural'] = Property_Area_Dummies['Rural']
loan_train['Property_Area_Semiurban'] = Property_Area_Dummies['Semiurban']

# TODO : Dropping Property_Area column as it is replaced with dummy columns
loan_train.drop('Property_Area', axis = 1, inplace = True)

loan_train.isnull().sum()
print(loan_train.dtypes)


# TODO : Display columns in Train Data
print(" Columns in the Train Data : \n", loan_train.columns)

# TODO : Dropping Loan_ID column from the Train data
loan_train.drop('Loan_ID', axis = 1, inplace = True)

numerical_continuous_data_column = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for column in numerical_continuous_data_column:
    sb.boxplot(x = column, y = 'Loan_Status', data = loan_train)
    plt.show()
# TODO : Display samples having outliers on ApplicantIncome
print(loan_train[loan_train['ApplicantIncome'] > 25000])

# TODO : Display samples having outliers on ApplicantIncome
print(loan_train[loan_train['CoapplicantIncome'] > 15000])

# TODO : Display Samples having outliers on LoanAmount
print(loan_train[loan_train['LoanAmount'] > 400])

# TODO : Rearranging Train Data columns in order to bring Loan_Status to the last of the DataFrame
loan_train = loan_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 10]]

# TODO : Let's do the feature correlation
loan_train_corr = loan_train.corr()
print(loan_train_corr)
# TODO : Visualizing correlation of features
# plt.figure(figsize = (16, 12))
# sb.heatmap(loan_train_corr, cmap = 'RdYlGn', annot = True, fmt = '.2%')
#plt.savefig('../images/correlation_of_features.png')

# TODO : To know whether the dataset is Balanced or Imbalanced
print(loan_train['Loan_Status'].value_counts())

# TODO : Correlation of Features
corr_with_loan_status = loan_train.corrwith(loan_train['Loan_Status'].apply(lambda x : {'Y' : 1, 'N' : 0}[x]))
print(corr_with_loan_status)

# Feature correlation with loan status
# plt.figure(figsize = (16, 4))
# sb.heatmap([corr_with_loan_status], cmap = 'RdYlGn', annot = True, fmt = '.2%')
#plt.savefig('../images/correlation_of_features_with_loan_status.png')

# TODO : To know the feature Importances
y = loan_train['Loan_Status'].apply(lambda x : {'Y' : 1, 'N' : 0}[x]).values
loan_train['Loan_Status']=loan_train['Loan_Status'].apply(lambda x : {'Y' : 1, 'N' : 0}[x])
loan_train.to_csv("summarizeddata.csv")
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(loan_train.iloc[:, :-1].values, y)

print("Percentage Importance of each features with respect to Loan_Status : ")
pd.Series(etc.feature_importances_*100, index = loan_train.columns[:-1])

prediction_features = pd.Series(etc.feature_importances_*100, index = loan_train.columns[:-1]).sort_values(ascending = False)
# TODO : Extracting Features name
prediction_features = prediction_features.index

prediction_features = prediction_features[:5]
print(prediction_features)
feature_columns = loan_train[prediction_features]
prediction_column = loan_train['Loan_Status']

X = feature_columns.values
y = prediction_column.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

logictic_model = LogisticRegression(max_iter = 200)
logictic_model.fit(X_train, y_train)

print("Training Accuracy : {}%".format(logictic_model.score(X_train, y_train) * 100))
print("Testing Accuracy  : {}%".format(logictic_model.score(X_test, y_test) * 100))

train_scores = []
test_scores = []
logistic_model_dict = {}
random_states = list(range(50))
for random_state in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    
    logictic_model = LogisticRegression(max_iter = 200)
    logictic_model.fit(X_train, y_train)
    
    train_score = logictic_model.score(X_train, y_train) * 100
    test_score = logictic_model.score(X_test, y_test) * 100
    
    logistic_model_dict[random_state] = {'Train Score' : train_score, 'Test Score' : test_score}
    train_scores.append(train_score)
    test_scores.append(test_score)

# plt.figure(figsize = (16, 8))
# plt.plot(random_states, train_scores, 'ro-')
# plt.plot(random_states, test_scores, 'go-')
# plt.xlabel('random_states', fontsize = 20)
# plt.ylabel('Scores', fontsize = 20)
# plt.title('Logistic Regression Model', fontsize = 30)
# # plt.ylim(0, 100)
# plt.legend(labels = ['Training Scores', 'Testing Scores'], fontsize = 20)
# plt.savefig('../images/logistic_model_performance.png')
# plt.show()