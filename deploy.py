import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os
import joblib

with open('scx.pkl', 'rb') as f:
    scx = pickle.load(f)
     
@st.cache(allow_output_mutation=True)
def PGs():
    PGA_model= joblib.load('models/Xgboost_ln(PGA).sav')
    PGV_model= joblib.load('models/Xgboost_ln(PGV).sav')
    return PGA_model,PGV_model
    
    
@st.cache(allow_output_mutation=True)
def call_models():
    T=[]
    models=[]  
    names=[]
    for root, dirs, files in os.walk('models/', topdown=False):
        for name in files:
            if name.find(model) != -1:
                if name.find('PG') == -1:             
                    T.append(float((name.replace('.sav','')).replace(f'{model}_ln(PSA=','').replace(')','')))
                    names.append(name)
                tuned_model= joblib.load(f'models/{name}')
                models.append(tuned_model)
    return models,T,names

    
model='Xgboost'
st.title("""
Ground motion model 
This app predicts the **geometric mean of Ground motion intensities** 
""")

st.sidebar.image("logo.png",width=30)
st.sidebar.title('Define your input')

Mw = st.sidebar.slider("Mw",min_value=4.0, value=6.0,max_value=7.6,step=0.1, help="Please enter a value between 4 and 7.6")
RJB = st.sidebar.slider("RJB",min_value=0, value=30,max_value=200,step=1, help="Please enter a value between 0 and 200 km")
Vs30 = st.sidebar.slider("Vs30",min_value=131, value=250,max_value=1380,step=1, help="Please enter a value between 131 and 1380 m/s2")
type = st.sidebar.radio(
    "Fault mechanism:",
    ('Reverse', 'strike-slip', 'Normal'))
if type=='Reverse':
    reverse=1
else:
    reverse=0
if type=='Normal':
    normal=1
else:
    normal=0
if type=='strike-slip':
    strike_slip=1
else:
    strike_slip=0
    
x=pd.DataFrame({'Mw':[Mw],'Vs30':[Vs30],'RJB':[RJB],'normal':[normal],'reverse':[reverse],'strike_slip':[strike_slip]})
st.title('Summary of your inputs:')
st.write(x)
st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
st.sidebar.markdown("---")

###############################################################
st.title('Outputs:')
PGA_model,PGV_model=PGs()
PGA=np.exp(PGA_model.predict(scx.transform(x))[0])/100
PGV=np.exp(PGV_model.predict(scx.transform(x))[0])
st.text('ln(PGA)= '+ str(np.round(PGA,2)) +'  m/s2')
st.text('ln(PGV)= '+ str(np.round(PGV,2)) +'  cm/s')

prediction=[] 
models,T,names=call_models()
st.write(names)
prediction=[]
for Model in models:
     prediction.append(np.exp(Model.predict(scx.transform(x))[0]))
            
fig, ax = plt.subplots(figsize=(8,2))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(T,prediction,color='k')
plt.xlabel('T (s)')
plt.ylabel(r'$PSA\ (cm/s^2)$')
plt.xlim(0.01,3.5)
plt.ylim(0,1000)
plt.grid(which='both')
plt.savefig('sprectra.png',dpi=600,bbox_inches='tight',pad_inches=0.05)
plt.gcf().subplots_adjust(bottom=0.15)

from PIL import Image
image = Image.open('sprectra.png')
st.image(image)

PSAs= pd.DataFrame([prediction],columns=T)
def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(PSAs)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='PSAs.csv',
    mime='text/csv',
)
