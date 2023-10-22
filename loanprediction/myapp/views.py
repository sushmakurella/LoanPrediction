from django.shortcuts import render,redirect
from .forms import ImageForm
from .models import ImageModel
import phe as paillier
import json
import pandas as pd
import pickle
import math
# Create your views here.
# def index(request):
#     return render(request,'index.html')

def home(request):
    return render(request, 'index.html')
def storeKeys():
	public_key, private_key = paillier.generate_paillier_keypair(n_length=32)
	keys={}
	keys['public_key'] = {'n': public_key.n}
	keys['private_key'] = {'p': private_key.p,'q':private_key.q}
	print(keys)
	with open('encrpkeys2.json', 'w') as file: 
		json.dump(keys, file)
def getKeys():
	with open('encrpkeys2.json', 'r') as file: 
		keys=json.load(file)
		pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
		priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
		return pub_key, priv_key 
# def calc(request):
#     if request.method == 'POST':
# 		name=request.POST['name']
#         age=request.POST['age']
#         a1=request.POST['gender']
#         a2=request.POST['married']
#         a3=request.POST['dep']
#         a4=request.POST['education']
#         a5=request.POST['self_employed']
#         a6=request.POST['annualincome']
#         a7=request.POST['cincome']
#         a8=request.POST['loanamount']
#         a9=request.POST['loanamountterm']
#         a10=request.POST['credithis']
#         a11=request.POST['property_Area']
# 		with open('encrpkeys2.json', 'r') as file: 
# 		    keys=json.load(file)
# 		    pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
# 		    priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
# 	return render(request, 'index.html')
'''def find(request):
	if request.method == "POST":
		name=request.POST['name']
		age=request.POST['age']
		v1=request.POST['gender']
		v2=request.POST['married']
		v3=int(request.POST['dep'])
		v4=request.POST['education']
		v5=request.POST['self_employed']
		v6=int(request.POST['annualincome'])
		v7=int(request.POST['cincome'])
		v8=int(request.POST['loanamount'])
		v9=int(request.POST['loanamountterm'])
		v10=int(request.POST['credithis'])
		v11=request.POST['property_Area']
		v12=0
		v1= 0 if(v1=='Male') else 1
		v2= 0 if(v2=='No') else 1
		v4= 0 if(v4=='UNderGraduate') else 1
		v5= 0 if(v5=='No') else 1
		if(v11=='Urban'):
			v11=0
		elif(v11=='Rural'):
			v11=1
		else:
			v11=2
		if(v3>3):
			v3=4
		# train_data['Gender']=train_data['Gender'].replace({'Male':0,'Female':1})
		# train_data['Married']=train_data['Married'].replace({'No':0,'Yes':1})
		# train_data['Education']=train_data['Education'].replace({'Graduate':1,'Not Graduate':0})
		# train_data['Self_Employed']=train_data['Self_Employed'].replace({'No':0,'Yes':1})
		# train_data['Property_Area']=train_data['Property_Area'].replace({'Urban':0,'Rural':1,'Semiurban':2})
		# train_data['Loan_Status']=train_data['Loan_Status'].replace({'N':0,'Y':1})
		# train_data['Dependents']=train_data['Dependents'].replace({'3+':4})
		with open('encrpkeys2.json', 'r') as file:
			keys=json.load(file)
			public_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
			priv_key=paillier.PaillierPrivateKey(public_key,keys['private_key']['p'],keys['private_key']['q'])
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
		#************
		a1=int(str(a1.ciphertext()))
		a2=int(str(a2.ciphertext()))
		a3=int(str(a3.ciphertext()))
		a4=int(str(a4.ciphertext()))
		a5=int(str(a5.ciphertext()))
		a6=int(str(a6.ciphertext()))
		a7=int(str(a7.ciphertext()))
		a8=int(str(a8.ciphertext()))
		a9=int(str(a9.ciphertext()))
		a10=int(str(a10.ciphertext()))
		a11=int(str(a11.ciphertext()))

		dt=[]
		lst=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
		cof=[[ 9.83916326e-21,1.42115851e-21,-2.47999311e-20,3.56947229e-20,1.01687148e-19,-3.09798568e-20,4.34032061e-20,7.13518885e-21,6.03465524e-20,2.06571937e-20,3.74646607e-20,-6.93733436e-20]]
		incp=[2.59904594e-38]
		dt.append(lst)
		# print(lst)
		# with open('logistic_regression_model.pkl', 'rb') as model_file:
		# 	logmodel = pickle.load(model_file)
		# pred=logmodel.predict(dt)
		# ans=(dt[0][0]*cof[0][0])+(dt[0][1]*cof[0][1])+(dt[0][2]*cof[0][2])+(dt[0][3]*cof[0][3])+(dt[0][4]*cof[0][4])+(dt[0][5]*cof[0][5])+(dt[0][6]*cof[0][6])+(dt[0][7]*cof[0][7])+(dt[0][8]*cof[0][8])+(dt[0][9]*cof[0][9])+(dt[0][10]*cof[0][10])+incp[0]
		res=priv_key.decrypt(int(ans))
		ans=(v1*cof[0][0])+(v2*cof[0][1])+(v3*cof[0][2])+(v4*cof[0][3])+(v5*cof[0][4])+(v6*cof[0][5])+(v7*cof[0][6])+(v8*cof[0][7])+(v9*cof[0][8])+(v10*cof[0][9])+(v11*cof[0][10])+incp[0]
		expr=math.exp(-ans)
		ans=1/(1+expr)
		# print(ans)
		# print(pred)
		# if(pred[0]==1):
		if(ans<=0.5):
			return render(request, 'congrat.html')
	return render(request, 'sorry.html')'''
def find(request):
	if request.method == "POST":
		name=request.POST['name']
		age=request.POST['age']
		v1=request.POST['gender']
		v2=request.POST['married']
		v3=int(request.POST['dep'])
		v4=request.POST['education']
		v5=request.POST['self_employed']
		v6=int(request.POST['annualincome'])
		v7=int(request.POST['cincome'])
		v8=int(request.POST['loanamount'])
		v9=int(request.POST['loanamountterm'])
		v10=int(request.POST['credithis'])
		v11=request.POST['property_Area']
		v12=0
		v1= 0 if(v1=='Male') else 1
		v2= 0 if(v2=='No') else 1
		v4= 0 if(v4=='UNderGraduate') else 1
		v5= 0 if(v5=='No') else 1
		if(v11=='Urban'):
			v11=1
			v12=0
		elif(v11=='Rural'):
			v11=0
			v12=0
		else:
			v11=0
			v12=1
		if(v3>3):
			v3=4
		with open('model.pkl', 'rb') as file:
			logmodel = pickle.load(file)
		lst=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12]
		dt=[]
		dt.append(lst)
		ans=logmodel.predict(dt)
		print(ans)
		if(ans==1):
			return render(request, 'congrat.html')
	return render(request, 'sorry.html')
def main(request):
	return render(request,'main.html')
def about(request):
	return render(request,'about.html')
def contact(request):
	return render(request,'contact.html')
def checkeligibility(request):
	return render(request,'loan.html')
		
		