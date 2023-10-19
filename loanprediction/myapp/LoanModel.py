import phe as paillier
import json
import pandas as pd
import pickle
# from sklearn.linear_model import LogisticRegression
def storeKeys():
	public_key, private_key = paillier.generate_paillier_keypair(n_length=32)
	keys={}
	keys['public_key'] = {'n': public_key.n}
	keys['private_key'] = {'p': private_key.p,'q':private_key.q}
	print(keys)
	with open('encrpkeys2.json', 'w') as file: 
		json.dump(keys, file)
#storeKeys()
def getKeys():
	with open('encrpkeys2.json', 'r') as file: 
		keys=json.load(file)
		pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
		priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
		return pub_key, priv_key 
def serializeData(public_key):
	encrypted_data={}
	train_data = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/train_u6lujuX_CVtuZ9i.csv")
	train_data=train_data.fillna(0)
	train_data['Gender']=train_data['Gender'].replace({'Male':0,'Female':1})
	train_data['Married']=train_data['Married'].replace({'No':0,'Yes':1})
	train_data['Education']=train_data['Education'].replace({'Graduate':1,'Not Graduate':0})
	train_data['Self_Employed']=train_data['Self_Employed'].replace({'No':0,'Yes':1})
	train_data['Property_Area']=train_data['Property_Area'].replace({'Urban':0,'Rural':1,'Semiurban':2})
	train_data['Loan_Status']=train_data['Loan_Status'].replace({'N':0,'Y':1})
	train_data['Dependents']=train_data['Dependents'].replace({'3+':4})
	train_data=train_data.drop(['Loan_ID'],axis=1)
	column_names = train_data.columns.tolist()
	print(column_names)
	name=['Gender','ge', 'Married','ma','Dependents','de','Education','ed', 'Self_Employed','se', 'ApplicantIncome','ap', 'CoapplicantIncome','co', 'LoanAmount','lo', 'Loan_Amount_Term','lot', 'Credit_History','cr','Property_Area','pr', 'Loan_Status']
	#print('#################################',len(name))
	# for i in range(len(name)):
	# 	print(i,'*****',name[i])
	df2=pd.DataFrame(columns=name)
	for index, row in train_data.iterrows():
		v1=int(row['Gender'])
		v2=int(row['Married'])
		v3=int(row['Dependents'])
		v4=int(row['Education'])
		v5=int(row['Self_Employed'])
		v6=int(row['ApplicantIncome'])
		v7=int(row['CoapplicantIncome'])
		v8=int(row['LoanAmount'])
		v9=int(row['Loan_Amount_Term'])
		v10=int(row['Credit_History'])
		v11=int(row['Property_Area'])
		a1=public_key.encrypt(v1)
		a2=public_key.encrypt(v2)
		a3=public_key.encrypt(v3)
		a4=public_key.encrypt(v4)
		a5=public_key.encrypt(v5)
		a6=public_key.encrypt(v6)
		a7=public_key.encrypt(v7)
		a8=public_key.encrypt(v8)
		a9=public_key.encrypt(v9)
		a10=public_key.encrypt(v10)
		a11=public_key.encrypt(v11)
		x1,y1=(str(a1.ciphertext()), a1.exponent)
		x2,y2=(str(a2.ciphertext()), a2.exponent)
		x3,y3=(str(a3.ciphertext()), a3.exponent)
		x4,y4=(str(a4.ciphertext()), a4.exponent)
		x5,y5=(str(a5.ciphertext()), a5.exponent)
		x6,y6=(str(a6.ciphertext()), a6.exponent)
		x7,y7=(str(a7.ciphertext()), a7.exponent)
		x8,y8=(str(a8.ciphertext()), a8.exponent)
		x9,y9=(str(a9.ciphertext()), a9.exponent)
		x10,y10=(str(a10.ciphertext()), a10.exponent)
		x11,y11=(str(a11.ciphertext()), a11.exponent)
		ls=row['Loan_Status']
		
		#print('#################################',len([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,ls]))
		df2.loc[len(df2)]=[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,ls]
	df2.to_csv("encryptedsaves3.csv")
# pub_key, priv_key = getKeys()
# serializeData(pub_key)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import phe as paillier
import json
train_data=pd.read_csv("encryptedsaves3.csv")
# X= train_data.drop(['s1','p1','r1','results'],axis=1)
X=train_data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
y = train_data['Loan_Status']
# print(X)
# print(y)
# train_data=pd.read_csv("emp.csv")
# X= train_data.drop('result',axis=1)
# y = train_data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=3)# random_state=3
'''logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#logmodel.fit(X,y)
print(logmodel.coef_)
print(logmodel.intercept_)
prediction= logmodel.predict(X_test)
print(prediction)
print(y_test.head())
accuracy = accuracy_score(y_test, prediction)
data = [0,0,0,1,0,5849,0,0,360,1]
# prediction= logmodel.predict(data)
# Pickle the model to a file
# with open('logistic_regression_model.pkl', 'wb') as model_file:
#     pickle.dump(logmodel, model_file)
# Load the pickled model'''
with open('logistic_regression_model.pkl', 'rb') as model_file:
    logmodel = pickle.load(model_file)
data = [[5132325828107003051,3892124869737499369,8012542509282533,1081922744201392948,3338514089537385108,2192964339753229640,3061289427997618936,3797293589316827850,1085908564160887396,1076333139637412075,1470495644499204620]]
prediction= logmodel.predict(X_test)
pred=logmodel.predict(data)
print(pred)
accuracy = accuracy_score(y_test, prediction)
print('$$$$$$$$$$$$$$$$$$$$$$',accuracy)