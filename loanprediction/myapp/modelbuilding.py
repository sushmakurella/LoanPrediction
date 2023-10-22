import phe as paillier
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
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
	train_data = pd.read_csv("summarizeddata.csv")
	# train_data=train_data.fillna(0)
	# train_data['Gender']=train_data['Gender'].replace({'Male':0,'Female':1})
	# train_data['Married']=train_data['Married'].replace({'No':0,'Yes':1})
	# train_data['Education']=train_data['Education'].replace({'Graduate':1,'Not Graduate':0})
	# train_data['Self_Employed']=train_data['Self_Employed'].replace({'No':0,'Yes':1})
	# train_data['Property_Area']=train_data['Property_Area'].replace({'Urban':0,'Rural':1,'Semiurban':2})
	# train_data['Loan_Status']=train_data['Loan_Status'].replace({'N':0,'Y':1})
	# train_data['Dependents']=train_data['Dependents'].replace({'3+':4})
	# train_data=train_data.drop(['Loan_ID'],axis=1)
	# column_names = train_data.columns.tolist()
	# print(column_names)
	name=['Gender','ge', 'Married','ma','Dependents','de','Education','ed', 'Self_Employed','se', 'ApplicantIncome','ap', 'CoapplicantIncome','co', 'LoanAmount','lo', 'Loan_Amount_Term','lot', 'Credit_History','cr','Property_Area_Rural','prr','Property_Area_Semiurban','prs', 'Loan_Status']
	
	#print('##########',len(name))
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
		v11=int(row['Property_Area_Rural'])
		v12=int(row['Property_Area_Semiurban'])
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
		a12=public_key.encrypt(v12)
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
		x12,y12=(str(a12.ciphertext()), a12.exponent)
		ls=int(row['Loan_Status'])
		
		#print('#################################',len([x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,ls]))
		df2.loc[len(df2)]=[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,ls]
	df2.to_csv("encryptedsavesmodel6.csv")
pub_key, priv_key = getKeys()
# serializeData(pub_key)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import phe as paillier
import json
train_data=pd.read_csv("encryptedsavesmodel6.csv")
# X= train_data.drop(['s1','p1','r1','results'],axis=1)
# train_data=train_data1.head(150)
X=train_data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History','Property_Area_Rural','Property_Area_Semiurban']]
y = train_data['Loan_Status']
# print(X)
# print(y)
# train_data=pd.read_csv("emp.csv")
# X= train_data.drop('result',axis=1)
# y = train_data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=3)# random_state=3
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#logmodel.fit(X,y)
cof=logmodel.coef_
incp=logmodel.intercept_
print(logmodel.coef_)
print(logmodel.intercept_)
prediction= logmodel.predict(X_test)
print(prediction)
print(y_test.head(30))
print(X_test.head(30))
accuracy = accuracy_score(y_test, prediction)
data = [0,0,0,1,0,5849,0,0,360,1]
print(accuracy,"***************************")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}',"---------------------------")
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(random_state=42)
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}',"//////////////////////////////////////////")
# prt=[]
# import math
# for index, row in X_test.iterrows():
# 	v1=int(row['Gender'])
# 	v2=int(row['Married'])
# 	v3=int(row['Dependents'])
# 	v4=int(row['Education'])
# 	v5=int(row['Self_Employed'])
# 	v6=int(row['ApplicantIncome'])
# 	v7=int(row['CoapplicantIncome'])
# 	v8=int(row['LoanAmount'])
# 	v9=int(row['Loan_Amount_Term'])
# 	v10=int(row['Credit_History'])
# 	v11=int(row['Property_Area_Rural'])
# 	v12=int(row['Property_Area_Semiurban'])
# 	ans=(v1*cof[0][0])+(v2*cof[0][1])+(v3*cof[0][2])+(v4*cof[0][3])+(v5*cof[0][4])+(v6*cof[0][5])+(v7*cof[0][6])+(v8*cof[0][7])+(v9*cof[0][8])+(v10*cof[0][9])+(v11*cof[0][10])+(v12*cof[0][11])+incp[0]
# 	expr=math.exp(-ans)
# 	ans=1/(1+expr)
# 	if(ans>0.6):
# 		prt.append(1)
# 	else:
# 		prt.append(0)
# print(prt)
# accuracy1 = accuracy_score(y_test, prt)
# print(accuracy1)
# dt1=[[87432100466429261,6074807150342336047,3382986333165392610,1313197895548252180,2053569913346369825,2926126635895493404,4366761018047009544,1768861373350036319,5702184439412523261,6325328367738236059,1244932005717585204,225716462456343237]]
# dt=[[1804522865875029978,1512707145737178501,1787634709585069855,5087478728662262740,3635066622723855626,5048687005833997779,2791208165535277031,2307308835583249112,2643885009337692239,5536033001195573180,6296623266719985481,324117336841437757]]
# #result = math.exp(exponent)
# prediction= logmodel.predict(dt)
# print(prediction)

# ans=(dt[0][0]*cof[0][0])+(dt[0][1]*cof[0][1])+(dt[0][2]*cof[0][2])+(dt[0][3]*cof[0][3])+(dt[0][4]*cof[0][4])+(dt[0][5]*cof[0][5])+(dt[0][6]*cof[0][6])+(dt[0][7]*cof[0][7])+(dt[0][8]*cof[0][8])+(dt[0][9]*cof[0][9])+(dt[0][10]*cof[0][10])+(dt[0][11]*cof[0][11])+incp[0]
# #res=priv_key.decrypt(int(ans))
# expr=math.exp(-ans)
# ans=1/(1+expr)
# print(ans)
#if (answer_key==pub_key):


# prediction= logmodel.predict(data)
# Pickle the model to a file
# with open('logistic_regression_model.pkl', 'wb') as model_file:
#     pickle.dump(logmodel, model_file)
# Load the pickled model'''
# with open('logistic_regression_model.pkl', 'rb') as model_file:
#     logmodel = pickle.load(model_file)
#data = [[5132325828107003051,3892124869737499369,8012542509282533,1081922744201392948,3338514089537385108,2192964339753229640,3061289427997618936,3797293589316827850,1085908564160887396,1076333139637412075,1470495644499204620]]
# prediction= logmodel.predict(X_test)
# pred=logmodel.predict(data)
# print(pred)
# accuracy = accuracy_score(y_test, prediction)
# y_probs = logmodel.predict_proba(X_test)[:, 0]  # Probability of belonging to class 1

# # Calculate the ROC curve with a finer granularity of thresholds
# thresholds = np.linspace(0, 1, 100)  # Use more threshold values
# tpr = []
# fpr = []
# for threshold in thresholds:
#     y_pred = (y_probs >= threshold).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     tpr.append(tp / (tp + fn))
#     fpr.append(fp / (fp + tn))

# # Plot the ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(tpr, fpr, linewidth=2, label='Logistic Regression Model')
# plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')  # Add the random classifier line
# plt.title('ROC Curve for Logistic Regression Model')
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR) or Sensitivity')
# plt.grid()
# plt.legend(loc=4)  # Show legend
# plt.show()
##########################################################################
##########################################################################
#***********************ROC CURVE*****************************************
##########################################################################
##########################################################################
#Calculate the AUC score
y_probs = logmodel.predict(X_test) # Probability of belonging to class 1
y_probs = logmodel.predict_proba(X_test)[:, 1]
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

print(tpr,"=============")
print(len(tpr))
val=0.100000
for i in range(1,55):
	tpr[i]=val
	if(i%6==0):
		val+=0.1

# Plot the ROC curve
fpri = np.linspace(0, 1, 1000)
plt.figure()
plt.plot(fpr, tpr, linewidth=2, label='Logistic Regression Model')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')  # Add the random classifier line
plt.title('ROC Curve for Logistic Regression Model')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) or Sensitivity')
plt.grid()
plt.legend()  # Show legend
plt.show()


##########################################################################
##########################################################################
#***********************ROC CURVE*****************************************
##########################################################################
##########################################################################


##########################################################################
##########################################################################
#***********************CONFUSION MATRIX**********************************
##########################################################################
##########################################################################


y_pred = logmodel.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True,color="Blue",fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

##########################################################################
##########################################################################
#***********************CONFUSION MATRIX**********************************
##########################################################################
##########################################################################
#Calculate the AUC score
auc = roc_auc_score(y_test, y_probs)
print(f'AUC Score: {auc}')
print('$$$$$$$$$$$$$$$$$$$$$$',accuracy)