# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 04:36:06 2022

@author: AmirHossein
"""

### Libraries
import pandas as pd
import numpy as np
import streamlit as st
from pickle import load
from bokeh.plotting import figure, show
################################################################
scx = load(open('scx.pkl', 'rb'))
scy = load(open('scy.pkl', 'rb'))
scpsa1 = load(open('scpsa1.pkl', 'rb'))
scpsa2 = load(open('scpsa2.pkl', 'rb'))

# load the model from disk
PGs = 'PGs.sav'
PSA1 = 'PSA1.sav'
PSA2 = 'PSA2.sav'
PGs = load(open(PGs, 'rb'))
PSA1 = load(open(PSA1, 'rb'))
PSA2 = load(open(PSA2, 'rb'))


st.write("""
# Ground motion model 
This app predicts the **geometric mean of PGA and PGV and PGD** 
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Magnitiude = st.sidebar.slider('M_w', 2.8, 7.6, 6.1)
    Vs30 = st.sidebar.slider('V_s30', 131.0, 1862.0, 345.942)
    RJB = st.sidebar.slider('R_jb', 0.0, 744.051174593, 10.0)
    Depth = st.sidebar.slider('Depth', 0.2, 95.0, 5.0)
    data = {'Magnitiude': Magnitiude,
            'Vs30': Vs30,
            'RJB': RJB,
            'Depth': Depth}

    Unscaled = pd.DataFrame(data, index=[0])

    df1=scx.transform(Unscaled)
    Scaled = pd.DataFrame(df1, index=[0])
    return Unscaled,Scaled

Inputs,Scaled_Inputs = user_input_features()

st.subheader('User Input parameters')
st.write(Inputs)

st.subheader('User scaled Input parameters')
st.write(Scaled_Inputs)

# import time
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.01)
#     my_bar.progress(percent_complete + 1)

st.subheader('Prediction of geometric means')
scaled_outputs=PGs.predict(Scaled_Inputs)
unscaled_outputs=scy.inverse_transform(scaled_outputs)
PGA_ln_G_mean=unscaled_outputs[0][0]
PGV_ln_G_mean=unscaled_outputs[0][1]
PGD_ln_G_mean=unscaled_outputs[0][2]

     
st.write('$\sqrt{PGA_1.PGA_2}=$', round(float(np.exp(PGA_ln_G_mean)),2))
st.write('$\sqrt{PGV_1.PGD_2}=$', round(float(np.exp(PGV_ln_G_mean)),2))
st.write('$\sqrt{PGD_1.PGD_2}=$', round(float(np.exp(PGD_ln_G_mean)),2))

st.subheader('Prediction of PSA1 and PSA2')
PSA1_out=PSA1.predict(Scaled_Inputs)
PSA2_out=PSA2.predict(Scaled_Inputs)

file = "Refined_data/T_dir.txt"
f=open(file,'r')
T_dir = f.read()
txt='/T.csv'
T=np.array(pd.read_csv(f'{T_dir}{txt}'))[0]
T= np.delete(T, (0), axis=0)

PSA1 = scpsa1.inverse_transform(PSA1_out)*float(np.exp(PGA_ln_G_mean))
PSA2 = scpsa2.inverse_transform(PSA2_out)*float(np.exp(PGA_ln_G_mean))

p1 = figure(
      title='This graph is predicted by GMM',
      x_axis_label='T',
      y_axis_label='PSA1',max_height=300,
    height_policy='max')

p1.line(T, PSA1[0], legend_label='Trend', line_width=2)
st.bokeh_chart(p1, use_container_width=True)

p2 = figure(
      title='This graph is predicted by GMM',
      x_axis_label='T',
      y_axis_label='PSA2',max_height=300,
    height_policy='max')

p2.line(T, PSA2[0], legend_label='Trend', line_width=2)
st.bokeh_chart(p2, use_container_width=True)

df1=pd.DataFrame([T,PSA1[0]])
df2=pd.DataFrame([T,PSA2[0]])

def convert_df(df1):
   return df1.to_csv().encode('utf-8')

def convert_df(df2):
   return df2.to_csv().encode('utf-8')

csv1 = convert_df(df1)
csv2 = convert_df(df2)

st.download_button(
   "Press to Download PSA1",
   csv1,
   "PSA1.csv",
   "text/csv",
   key='download-csv'
)

st.download_button(
   "Press to Download PSA2",
   csv2,
   "PSA2.csv",
   "text/csv",
   key='download-csv'
)