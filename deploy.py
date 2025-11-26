import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os
import joblib
from PIL import Image

# ===================== Load scaler & stds ======================

with open('scx.pkl', 'rb') as f:
    scx = pickle.load(f)

# stds.csv has columns: ID, Sigma (intra), Tau (inter), Phi (total)
stds_df = pd.read_csv("stds.csv")

# ===================== Load models ======================

@st.cache_resource
def PGs():
    PGA_model = joblib.load('models/Xgboost_ln(PGA).sav')
    PGV_model = joblib.load('models/Xgboost_ln(PGV).sav')
    return PGA_model, PGV_model

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

# ===================== Streamlit UI ======================

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

# ---- Show Scaled Inputs (validation) ----
scaled_x = scx.transform(x)[0]  # array of length 6

scaled_df = pd.DataFrame({
    "Feature": ["Mw", "Vs30", "RJB", "normal", "reverse", "strike_slip"],
    "Original": x.iloc[0].values,
    "Scaled": scaled_x,
})

st.subheader("🔎 Scaled Inputs (MinMax Output)")
st.dataframe(scaled_df.style.format({"Original": "{:.3f}", "Scaled": "{:.5f}"}))

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

# ===================== Helpers ======================

def preprocess_batch(df):
    return scx.transform(df[['Mw', 'Vs30', 'RJB', 'normal', 'reverse', 'strike_slip']])

def run_batch(df):
    PGA_model, PGV_model = PGs()
    models, T, _ = call_models()

    X = preprocess_batch(df)
    df = df.copy()
    # ln(IM) -> IM
    df['PGA'] = np.exp(PGA_model.predict(X))   # cm/s²
    df['PGV'] = np.exp(PGV_model.predict(X))   # cm/s

    for mdl, t in zip(models, T):
        df[f'PSA_{t}s'] = np.exp(mdl.predict(X))   # cm/s²

    return df, sorted(T)

# ===================== Main Output Section ======================

st.title('Outputs:')

if uploaded_file is not None:
    # -------- BATCH MODE --------
    df_in = pd.read_csv(uploaded_file)
    df_out, T_list = run_batch(df_in)

    st.subheader("📊 Batch Predictions")
    st.dataframe(df_out)

    # Plot all PSA curves
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
    # -------- SINGLE RECORD MODE --------
    PGA_model, PGV_model = PGs()
    X_single = scx.transform(x)

    # ln(IM)
    lnPGA = PGA_model.predict(X_single)[0]
    lnPGV = PGV_model.predict(X_single)[0]

    # IM in linear units
    PGA = np.exp(lnPGA)   # cm/s²
    PGV = np.exp(lnPGV)   # cm/s

    # Get TOTAL std (Phi) for PGA and PGV in ln units
    Phi_PGA = stds_df.loc[stds_df["ID"] == "ln(PGA)", "Phi"].values[0]
    Phi_PGV = stds_df.loc[stds_df["ID"] == "ln(PGV)", "Phi"].values[0]

    # Correct ±1 std ranges:
    # Upper = exp(ln(IM) + Phi), Lower = exp(ln(IM) - Phi)
    PGA_upper = np.exp(lnPGA + Phi_PGA)
    PGA_lower = np.exp(lnPGA - Phi_PGA)
    PGV_upper = np.exp(lnPGV + Phi_PGV)
    PGV_lower = np.exp(lnPGV - Phi_PGV)

    st.text(f'PGA = {np.round(PGA, 2)} cm/s² (+- {np.round(PGA_upper - PGA, 2)})')
    st.text(f'PGV = {np.round(PGV, 2)} cm/s (+- {np.round(PGV_upper - PGV, 2)})')

    # PSA predictions for this single input
    models, T, names = call_models()

    lnPSA_list = []
    PSA_list = []
    Phi_list = []

    for t, Model in zip(T, models):
        lnPSA = Model.predict(X_single)[0]       # ln(PSA[cm/s²])
        lnPSA_list.append(lnPSA)
        PSA_list.append(np.exp(lnPSA))
        # total std (Phi) for that period
        Phi_t = stds_df.loc[stds_df["ID"] == f"ln(PSA={t})", "Phi"].values[0]
        Phi_list.append(Phi_t)

    PSAs = pd.DataFrame()
    PSAs['T'] = T
    PSAs['lnPSA'] = lnPSA_list
    PSAs['PSAs'] = PSA_list                 # median PSA in cm/s²
    PSAs['Phi'] = Phi_list                  # total std in ln units

    # Correct upper/lower envelopes
    PSAs['Upper'] = np.exp(PSAs['lnPSA'] + PSAs['Phi'])
    PSAs['Lower'] = np.exp(PSAs['lnPSA'] - PSAs['Phi'])

    PSAs.sort_values(by=["T"], inplace=True)
    PSAs.reset_index(drop=True, inplace=True)

    # ---- Plot response spectrum with ±1Φ ----
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Median curve
    ax.plot(PSAs['T'], PSAs['PSAs'], color='k', label="Median")

    # Shaded ± total std (Phi)
    ax.fill_between(
        PSAs['T'], PSAs['Lower'], PSAs['Upper'],
        color='gray', alpha=0.3, label=r'$\pm 1\Phi_{\mathrm{total}}$'
    )

    plt.xlabel('T (s)')
    plt.ylabel(r'$PSA\ (cm/s^2)$')
    plt.xlim(0.01, 3.5)
    plt.ylim(0.1, 1000)  # avoid zero for log scale
    plt.grid(which='both')
    plt.savefig('sprectra.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.gcf().subplots_adjust(bottom=0.15)

    image = Image.open('sprectra.png')
    st.image(image)

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(PSAs)
    st.download_button("📥 Download PSA data as CSV", data=csv, file_name='PSAs.csv', mime='text/csv')

# ===================== Plot Tau / Sigma / Phi over IMs ======================

fig, ax = plt.subplots(figsize=(18, 4))

ax.plot(stds_df.ID, stds_df['Tau'],   label=r'$\tau$ (inter-event)', marker='o')
ax.plot(stds_df.ID, stds_df['Sigma'], label=r'$\sigma$ (intra-event)', marker='s')
ax.plot(stds_df.ID, stds_df['Phi'],   label=r'$\Phi$ (total)', marker='^')

plt.setp(ax.get_xticklabels(), rotation=90)

ax.set_xlabel("IMs")
ax.set_ylabel("Value (ln-units)")
ax.set_title(r"$\tau$ (inter), $\sigma$ (intra), and $\Phi$ (total) for IMs")
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
