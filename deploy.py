# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import pickle
# import os
# import joblib

# with open('scx.pkl', 'rb') as f:
#     scx = pickle.load(f)
     
# @st.cache_resource
# def PGs():
#     PGA_model= joblib.load('models/Xgboost_ln(PGA).sav')
#     PGV_model= joblib.load('models/Xgboost_ln(PGV).sav')
#     return PGA_model,PGV_model
    
    
# @st.cache_resource
# def call_models():
#     T=[]
#     models=[]  
#     names=[]
#     for root, dirs, files in os.walk('models/', topdown=False):
#         for name in files:
#             if name.find(model) != -1:
#                 if name.find('PG') == -1:             
#                     T.append(float((name.replace('.sav','')).replace(f'{model}_ln(PSA=','').replace(')','')))
#                     names.append(name)
#                     tuned_model= joblib.load(f'models/{name}')
#                     models.append(tuned_model)
#     return models,T,names

    
# model='Xgboost'
# st.title("""
# Ground motion model 
# This app predicts the **geometric mean of ground motion intensities** 
# """)

# st.sidebar.image("logo.png",width=30)
# st.sidebar.title('Define your input')

# Mw = st.sidebar.slider("Mw",min_value=4.0, value=6.0,max_value=7.6,step=0.1, help="Please enter a value between 4 and 7.6")
# RJB = st.sidebar.slider("RJB",min_value=0, value=30,max_value=200,step=1, help="Please enter a value between 0 and 200 km")
# Vs30 = st.sidebar.slider("Vs30",min_value=131, value=250,max_value=1380,step=1, help="Please enter a value between 131 and 1380 m/s2")
# type = st.sidebar.radio(
#     "Fault mechanism:",
#     ('Reverse', 'strike-slip', 'Normal'))
# if type=='Reverse':
#     reverse=1
# else:
#     reverse=0
# if type=='Normal':
#     normal=1
# else:
#     normal=0
# if type=='strike-slip':
#     strike_slip=1
# else:
#     strike_slip=0
    
# x=pd.DataFrame({'Mw':[Mw],'Vs30':[Vs30],'RJB':[RJB],'normal':[normal],'reverse':[reverse],'strike_slip':[strike_slip]})
# st.title('Summary of your inputs:')
# st.write(x)
# st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
# st.sidebar.markdown("---")

# ###############################################################
# st.title('Outputs:')
# PGA_model,PGV_model=PGs()
# PGA=np.exp(PGA_model.predict(scx.transform(x))[0])
# PGV=np.exp(PGV_model.predict(scx.transform(x))[0])
# st.text('PGA= '+ str(np.round(PGA,2)) +'  cm/s2')
# st.text('PGV= '+ str(np.round(PGV,2)) +'  cm/s')

# prediction=[] 
# models,T,names=call_models()

# prediction=[]
# for Model in models:
#      prediction.append(np.exp(Model.predict(scx.transform(x))[0]))

# PSAs= pd.DataFrame()
# PSAs['PSAs']=prediction
# PSAs['T']=T
# PSAs.sort_values(by=["T"], inplace = True) 
# PSAs.reset_index(drop=True,inplace=True)

# fig, ax = plt.subplots(figsize=(8,2))
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.plot(PSAs['T'],PSAs['PSAs'],color='k')
# plt.xlabel('T (s)')
# plt.ylabel(r'$PSA\ (cm/s^2)$')
# plt.xlim(0.01,3.5)
# plt.ylim(0,1000)
# plt.grid(which='both')
# plt.savefig('sprectra.png',dpi=600,bbox_inches='tight',pad_inches=0.05)
# plt.gcf().subplots_adjust(bottom=0.15)

# from PIL import Image
# image = Image.open('sprectra.png')
# st.image(image)

# def convert_df(df):
#     return df.to_csv().encode('utf-8')
# csv = convert_df(PSAs)

# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='PSAs.csv',
#     mime='text/csv',
# )



import pandas as pd
import numpy as np
import pickle, joblib, os
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Load your scaler and models
with open('scx.pkl','rb') as f:
    scx = pickle.load(f)

@st.cache_resource
def load_models():
    PGA_model = joblib.load('models/Xgboost_ln(PGA).sav')
    PGV_model = joblib.load('models/Xgboost_ln(PGV).sav')
    return PGA_model, PGV_model

@st.cache_resource
def load_psa_models():
    models, T, names = [], [], []
    for root, dirs, files in os.walk('models/'):
        for name in files:
            if 'Xgboost' in name and 'PG' not in name and name.endswith('.sav'):
                T.append(float(name.split('PSA=')[1].split(')')[0]))
                models.append(joblib.load(os.path.join(root,name)))
                names.append(name)
    return models, T, names

st.title("Ground motion model – single & batch mode")

# Sidebar: single record inputs as before...
# [Your existing widgets here]

# Add file uploader for batch processing
uploaded_file = st.sidebar.file_uploader(
    "Or upload CSV for batch prediction", type="csv"
)

PGA_model, PGV_model = load_models()
psa_models, psa_Ts, psa_names = load_psa_models()

def preprocess(df):
    df = df[['Mw','Vs30','RJB','normal','reverse','strike_slip']]
    return scx.transform(df)

def run_batch(df):
    X = preprocess(df)
    df_out = df.copy()
    df_out['PGA'] = np.exp(PGA_model.predict(X))
    df_out['PGV'] = np.exp(PGV_model.predict(X))
    # PSA predictions
    for mdl, T in zip(psa_models, psa_Ts):
        df_out[f'PSA_{T}s'] = np.exp(mdl.predict(X))
    return df_out

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_res = run_batch(df)

    st.subheader("Batch Predictions")
    st.dataframe(df_res)

    csv = df_res.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions CSV", data=csv,
                       file_name="batch_predictions.csv", mime="text/csv")

else:
    # Existing single-record predictions for sidebar inputs
    x = pd.DataFrame({...})  # your existing single-row DataFrame creation
    PGA = np.exp(PGA_model.predict(scx.transform(x))[0])
    PGV = np.exp(PGV_model.predict(scx.transform(x))[0])
    st.subheader("Single record results")
    st.write(f"PGA = {PGA:.2f} cm/s²")
    st.write(f"PGV = {PGV:.2f} cm/s")

    # PSA plot
    psa_vals = [np.exp(m.predict(scx.transform(x))[0]) for m in psa_models]
    psa_df = pd.DataFrame({'T': psa_Ts, 'PSA': psa_vals}).sort_values('T')
    fig, ax = plt.subplots()
    ax.loglog(psa_df['T'], psa_df['PSA'], '-k')
    ax.set(xlabel='T (s)', ylabel='PSA (cm/s²)', xlim=(0.01,3.5))
    ax.grid(True, which='both')
    st.pyplot(fig)

    # Download PSA CSV
    csv2 = psa_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download PSA CSV", data=csv2,
                       file_name="PSAs.csv", mime="text/csv")
