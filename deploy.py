import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os
from PIL import Image


import onnxruntime as ort  # NEW

def load_css():
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Hides the Streamlit hamburger menu and footer for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Header Styling */
        h1 {
            color: #2c3e50;
            font-weight: 700;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
        }
        h2, h3 {
            color: #34495e;
        }

        /* Metric Card Styling */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2980b9;
        }
        .metric-label {
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .metric-delta {
            font-size: 0.9rem;
            color: #95a5a6;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f7f9fc;
            border-right: 1px solid #e6e6e6;
        }
        
        /* Button Styling */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #2980b9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)


def load_onnx_model(path: str):
    """
    Load an ONNX model and return a callable predict(X) -> 1D np.array.

    X must be a 2D array (n_samples, n_features).
    """
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def predict(X):
        X_arr = np.asarray(X, dtype=np.float32)
        # sess.run returns a list of outputs; we assume a single output
        y = sess.run(None, {input_name: X_arr})[0]
        return y.reshape((X_arr.shape[0],))

    return predict

# ===================== Load scaler & stds ======================

with open('scx.pkl', 'rb') as f:
    scx = pickle.load(f)

# stds.csv has columns: ID, Sigma (intra), Tau (inter), Phi (total)
stds_df = pd.read_csv("stds.csv")

# ===================== Load models ======================

@st.cache_resource
def PGs():
    """
    Load ONNX models for ln(PGA) and ln(PGV).
    Returned objects are callables: y = model(X).
    """
    PGA_model = load_onnx_model('onnx_models/Xgboost_ln(PGA).onnx')
    PGV_model = load_onnx_model('onnx_models/Xgboost_ln(PGV).onnx')
    return PGA_model, PGV_model

@st.cache_resource
def call_models():
    """
    Load ONNX PSA models.

    Expects files like:
        onnx_models/Xgboost_ln(PSA=0.10).onnx
        onnx_models/Xgboost_ln(PSA=0.20).onnx
        ...
    """
    T = []
    models = []
    names = []
    model = "Xgboost"
    for root, dirs, files in os.walk('onnx_models/', topdown=False):
        for name in files:
            if name.endswith(".onnx") and name.find(model) != -1 and name.find('PG') == -1:
                # Extract the period from e.g. "Xgboost_ln(PSA=0.20).onnx"
                period_str = (
                    name.replace(".onnx", "")
                        .replace(f"{model}_ln(PSA=", "")
                        .replace(")", "")
                )
                T.append(float(period_str))

                full_path = os.path.join(root, name)
                tuned_model = load_onnx_model(full_path)
                models.append(tuned_model)
                names.append(name)

    return models, T, names


# ===================== Streamlit UI ======================
load_css()

st.title("Ground Motion Model")
st.markdown("This app predicts the **geometric mean of ground motion intensities** using a machine learning based approach.")
st.markdown("---")

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

# st.subheader("🔎 Scaled Inputs (MinMax Output)")
# st.dataframe(scaled_df.style.format({"Original": "{:.3f}", "Scaled": "{:.5f}"}))

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
        label="📥 Download CSV Template",
        data=file,
        file_name="example_batch_input.csv",
        mime="text/csv"
    )

uploaded_file = st.sidebar.file_uploader("Or upload CSV for batch prediction", type='csv')

# Sidebar footer
st.sidebar.markdown("Made by [Amirhossein Mohammadi](https://www.linkedin.com/in/amir-hossein-mohammadi-86729957/)")
st.sidebar.markdown("---")

# Title - Summary of Input
with st.container():
    st.subheader("Your Input Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moment Magnitude (Mw)", f"{Mw}")
    col2.metric("Distance (RJB)", f"{RJB} km")
    col3.metric("Vs30", f"{Vs30} m/s")
    col4.metric("Mechanism", type)
    # st.write(x) # Debug view suppressed for cleaner UI

# ===================== Helpers ======================

def preprocess_batch(df):
    return scx.transform(df[['Mw', 'Vs30', 'RJB', 'normal', 'reverse', 'strike_slip']])

def run_batch(df):
    PGA_model, PGV_model = PGs()
    models, T, _ = call_models()

    X = preprocess_batch(df)
    df = df.copy()

    # ln(IM) -> IM
    lnPGA = PGA_model(X)          # shape (n_samples,)
    lnPGV = PGV_model(X)
    df['PGA'] = np.exp(lnPGA)     # cm/s²
    df['PGV'] = np.exp(lnPGV)     # cm/s

    for mdl, t in zip(models, T):
        lnPSA = mdl(X)
        df[f'PSA_{t}s'] = np.exp(lnPSA)   # cm/s²

    return df, sorted(T)

# ===================== Main Output Section ======================

st.title('Outputs:')

if uploaded_file is not None:
    # -------- BATCH MODE --------
    # -------- BATCH MODE --------
    # Helper to try robust reading strategies
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16"]
    df_in = None
    
    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)  # Reset buffer position before each attempt
            df_in = pd.read_csv(uploaded_file, sep=None, engine='python', encoding=encoding)
            break
        except Exception:
            continue
            
    if df_in is None:
        st.error("Could not read the file. Please check encoding (try saving as standard CSV UTF-8).")
        st.stop()
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
    lnPGA = PGA_model(X_single)[0]
    lnPGV = PGV_model(X_single)[0]


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

    # Display Results in Cards using standard Streamlit Metrics
    st.markdown("### 📊 Prediction Results")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.info(f"**PGA** (Peak Ground Acceleration)")
        st.metric(
            label="Median PGA",
            value=f"{np.round(PGA, 2)} cm/s²",
            delta=f"± {np.round(PGA_upper - PGA, 2)}"
        )
    
    with colB:
        st.info(f"**PGV** (Peak Ground Velocity)")
        st.metric(
            label="Median PGV",
            value=f"{np.round(PGV, 2)} cm/s",
            delta=f"± {np.round(PGV_upper - PGV, 2)}"
        )

    # PSA predictions for this single input
    models, T, names = call_models()

    lnPSA_list = []
    PSA_list = []
    Phi_list = []

    for t, Model in zip(T, models):
        lnPSA = Model(X_single)[0]              # ln(PSA[cm/s²])
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



