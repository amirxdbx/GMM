# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import pickle
# import os
# import joblib
# import io


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

# uploaded_file = st.sidebar.file_uploader(
#     "Or upload a CSV for batch prediction", type="csv"
# )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os
import joblib
from PIL import Image
import io

# Load scaler
with open('scx.pkl', 'rb') as f:
    scx = pickle.load(f)

# Load PGA and PGV models
@st.cache_resource
def PGs():
    PGA_model = joblib.load('models/Xgboost_ln(PGA).sav')
    PGV_model = joblib.load('models/Xgboost_ln(PGV).sav')
    return PGA_model, PGV_model

# Load PSA models
@st.cache_resource
def call_models():
    T = []
    models = []
    names = []
    for root, dirs, files in os.walk('models/', topdown=False):
        for name in files:
            if name.find(model) != -1 and name.find('PG') == -1:
                T.append(float((name.replace('.sav', '')).replace(f'{model}_ln(PSA=', '').replace(')', '')))
                tuned_model = joblib.load(f'models/{name}')
                models.append(tuned_model)
                names.append(name)
    return models, T, names

model = 'Xgboost'

# Streamlit Title
st.title("Ground Motion Model")
st.write("This app predicts the **geometric mean of ground motion intensities**.")

# Sidebar - Input
st.sidebar.image("logo.png", width=30)
st.sidebar.title('Define your input')

Mw = st.sidebar.slider("Mw", min_value=4.0, value=6.0, max_value=7.6, step=0.1)
RJB = st.sidebar.slider("RJB", min_value=0, value=30, max_value=200, step=1)
Vs30 = st.sidebar.slider("Vs30", min_value=131, value=250, max_value=1380, step=1)
type = st.sidebar.radio("Fault mechanism:", ('Reverse', 'strike-slip', 'Normal'))

reverse = 1 if type == 'Reverse' else 0
normal = 1 if type == 'Normal' else 0
strike_slip = 1 if type == 'strike-slip' else 0

x = pd.DataFrame({
    'Mw': [Mw],
    'Vs30': [Vs30],
    'RJB': [RJB],
    'normal': [normal],
    'reverse': [reverse],
    'strike_slip': [strike_slip]
})

# Sidebar - CSV Upload
st.sidebar.markdown("### 📥 Batch Prediction Instructions")

st.sidebar.markdown("""
You can perform predictions for multiple records by uploading a CSV or Excel file.

Each row must include the following columns:

- **Mw** (e.g., 4.0 to 7.6)  
- **Vs30** (e.g., 131 to 1380 m/s)  
- **RJB** (e.g., 0 to 200 km)  
- **normal**, **reverse**, **strike_slip** (only one should be 1 per row)
""")

with open("example_batch_input.csv", "rb") as file:
    st.sidebar.download_button(
        label="📥 Download Excel Template",
        data=file,
        file_name="example_batch_input.csv",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    
uploaded_file = st.sidebar.file_uploader("Or upload CSV for batch prediction", type='csv')

# Sidebar footer

st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
st.sidebar.markdown("---")

# Title - Summary of Input
st.title("Summary of Your Inputs:")
st.write(x)

# ===================== Preprocessing and Batch Prediction ======================

def preprocess_batch(df):
    return scx.transform(df[['Mw', 'Vs30', 'RJB', 'normal', 'reverse', 'strike_slip']])

def run_batch(df):
    PGA_model, PGV_model = PGs()
    models, T, _ = call_models()

    X = preprocess_batch(df)
    df = df.copy()
    df['PGA'] = np.exp(PGA_model.predict(X))
    df['PGV'] = np.exp(PGV_model.predict(X))

    for mdl, t in zip(models, T):
        df[f'PSA_{t}s'] = np.exp(mdl.predict(X))

    return df, sorted(T)

# ===================== Main Output Section ======================

st.title('Outputs:')

if uploaded_file is not None:
    df_in = pd.read_csv(uploaded_file)
    df_out, T_list = run_batch(df_in)

    st.subheader("📊 Batch Predictions")
    st.dataframe(df_out)

   # Optional: Plot all PSA curves
    fig, ax = plt.subplots(figsize=(8, 4))
     
    for idx, row in df_out.iterrows():
         tsa = [row[f'PSA_{t}s'] for t in T_list]
         ax.loglog(T_list, tsa, alpha=0.6, label=f'Record {idx+1}')
     
    ax.set(xlabel='T (s)', ylabel='PSA (cm/s²)', xlim=(0.01, 3.5))
    ax.grid(True, which='both')
    ax.set_title("PSA Spectra for All Records")
    ax.legend(loc='best', fontsize='small', ncol=2)
     
    st.pyplot(fig)

    # Download option
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download batch results as CSV", data=csv, file_name="batch_results.csv", mime="text/csv")

else:
    PGA_model, PGV_model = PGs()
    PGA = np.exp(PGA_model.predict(scx.transform(x))[0])
    PGV = np.exp(PGV_model.predict(scx.transform(x))[0])
    st.text(f'PGA = {np.round(PGA, 2)} cm/s²')
    st.text(f'PGV = {np.round(PGV, 2)} cm/s')

    prediction = []
    models, T, names = call_models()

    for Model in models:
        prediction.append(np.exp(Model.predict(scx.transform(x))[0]))

    PSAs = pd.DataFrame()
    PSAs['PSAs'] = prediction
    PSAs['T'] = T
    PSAs.sort_values(by=["T"], inplace=True)
    PSAs.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(PSAs['T'], PSAs['PSAs'], color='k')
    plt.xlabel('T (s)')
    plt.ylabel(r'$PSA\ (cm/s^2)$')
    plt.xlim(0.01, 3.5)
    plt.ylim(0, 1000)
    plt.grid(which='both')
    plt.savefig('sprectra.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.gcf().subplots_adjust(bottom=0.15)

    image = Image.open('sprectra.png')
    st.image(image)

    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(PSAs)
    st.download_button("📥 Download PSA data as CSV", data=csv, file_name='PSAs.csv', mime='text/csv')


# Load stds.csv
stds_df = pd.read_csv("stds.csv")

# Plot Tau, Sigma, Phi
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stds_df.ID, stds_df['Tau'], label=r'$\tau$', marker='o')
ax.plot(stds_df.ID, stds_df['Sigma'], label=r'$\sigma$', marker='s')
ax.plot(stds_df.ID, stds_df['Phi'], label=r'$\phi$', marker='^')

ax.set_xlabel("Item ID")
ax.set_ylabel("Value")
ax.set_title("τ, σ, and ϕ for IMs")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Download button for stds.csv
with open("stds.csv", "rb") as file:
    st.download_button(
        label="📥 Download Standard deviations",
        data=file,
        file_name="stds.csv",
        mime="text/csv"
    )



