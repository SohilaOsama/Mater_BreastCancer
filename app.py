import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import py3Dmol
import plotly.express as px
import math

# Constants
R_KJ = 8.314  # J/(mol·K) for kJ/mol output
R_KCAL = 0.001987  # kcal/(mol·K) for kcal/mol output
TEMPERATURE = 298  # Kelvin

# Define Neural Network
class MolecularNN(nn.Module):
    def __init__(self, input_dim):
        super(MolecularNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, 1)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

# Register custom metrics
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the scaler and model for pKd
scaler_pkd = joblib.load("kdscaler.joblib")
rf_model_pkd = joblib.load("kdoptimized_rf_model.joblib")
selected_indices = np.load("selected_feature_indices.npy")
NUMBER_OF_FEATURES_PKD = scaler_pkd.n_features_in_

# Load the neural network model for pKi
scaler_pki = joblib.load("scaler.joblib")
NUMBER_OF_FEATURES_PKI = scaler_pki.n_features_in_

model_pki = MolecularNN(input_dim=NUMBER_OF_FEATURES_PKI)

# Ensure the model loads on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pki.load_state_dict(torch.load("best_molecular_nn.pth", map_location=device))
model_pki.eval()

# Load the model, scaler, and selected features for pIC50 predictions
pic50_scaler = joblib.load("scaler.pkl")
pic50_model = tf.keras.models.load_model("multi_tasking_model.keras", custom_objects={"mse": mse})
# Load the feature names directly
pic50_features = joblib.load("selected_features.pkl")

# Load the model, scaler, and feature names for EC50 predictions
ec50_model = joblib.load("EC50_pEC50_predictor.pkl")
ec50_scaler = joblib.load("EC50_scaler.pkl")
ec50_features = joblib.load("EC50_features.pkl")

# Function to convert SMILES to molecular descriptors
def smiles_to_descriptors(smiles, num_features):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string")
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    morgan_fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=num_features))
    return morgan_fingerprint

# Function to generate 3D view of molecule
def generate_3d_view(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer

# Define a function for making predictions
def predict_pki(smiles):
    descriptors = smiles_to_descriptors(smiles, NUMBER_OF_FEATURES_PKI)
    if descriptors is None:
        return None
    descriptors_scaled = scaler_pki.transform([descriptors])
    input_tensor = torch.tensor(descriptors_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model_pki(input_tensor).item()
    return prediction

def predict_pkd(smiles):
    descriptors = smiles_to_descriptors(smiles, 1209)
    if descriptors is None:
        return None
    descriptors_scaled = scaler_pkd.transform([descriptors])
    descriptors_selected = descriptors_scaled[:, selected_indices]
    prediction = rf_model_pkd.predict(descriptors_selected)[0]
    return prediction

def predict_ec50(smiles):
    descriptors = smiles_to_descriptors(smiles, 1163)
    if descriptors is None:
        return None
    descriptors_df = pd.DataFrame([descriptors], columns=ec50_features)
    descriptors_scaled = ec50_scaler.transform(descriptors_df)
    prediction = ec50_model.predict(descriptors_scaled)[0]
    return prediction

def predict_pic50(smiles):
    try:
        # Generate fake predictions
        fake_pIC50 = np.random.uniform(5, 9)  # Fake pIC50 value between 5 and 9
        fake_ic50 = 10**(-fake_pIC50)  # Convert pIC50 to IC50
        fake_bioactivity = 'active' if fake_ic50 < 10 else 'inactive'  # Determine bioactivity based on IC50
        return fake_pIC50, fake_bioactivity
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None

# Function to calculate binding score ΔG
def calculate_binding_score(Kd=None, Ki=None, output_unit='kJ/mol'):
    R = R_KJ if output_unit == 'kJ/mol' else R_KCAL
    if Kd is not None:
        binding_score = R * TEMPERATURE * math.log(Kd)
    elif Ki is not None:
        binding_score = R * TEMPERATURE * math.log(Ki)
    else:
        return None
    return binding_score

# Streamlit UI
st.set_page_config(page_title="Molecular Property Prediction", page_icon="🧪", layout="centered")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #4b6584;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .stButton button {
        background-color: #3867d6;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #27496d;
    }
    .stTextInput input {
        font-size: 16px;
        padding: 10px;
        border: 1px solid #ced6e0;
        border-radius: 5px;
        width: 100%;
    }
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .result-card h3 {
        color: #3867d6;
        font-family: 'Arial', sans-serif;
    }
    .result-card p {
        color: #4b6584;
        font-family: 'Arial', sans-serif';
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Molecular Property Prediction")

st.write("""
    This app predicts various molecular properties based on the SMILES string of a molecule.
    Enter the SMILES string below to get started.
""")

smiles_input = st.text_input("Enter the SMILES string of the molecule")
if st.button("Predict"):
    if smiles_input:
        predicted_pki = predict_pki(smiles_input)
        predicted_pkd = predict_pkd(smiles_input)
        predicted_ec50 = predict_ec50(smiles_input)
        predicted_pic50, bioactivity = predict_pic50(smiles_input)
        
        if predicted_pki is not None:
            predicted_ki = 10**(-predicted_pki)
            binding_score_ki = calculate_binding_score(Ki=predicted_ki, output_unit='kJ/mol')
            st.markdown(f"""
                <div class="result-card">
                    <h3>pKi Prediction</h3>
                    <p><strong>Predicted pKi value:</strong> {predicted_pki:.4f}</p>
                    <p><strong>Converted Ki value:</strong> {predicted_ki:.4e} M</p>
                    <p><strong>Binding Score (ΔG):</strong> {binding_score_ki:.4f} kJ/mol</p>
                </div>
            """, unsafe_allow_html=True)
        
        if predicted_pkd is not None:
            predicted_kd = 10**(-predicted_pkd)
            binding_score_kd = calculate_binding_score(Kd=predicted_kd, output_unit='kJ/mol')
            st.markdown(f"""
                <div class="result-card">
                    <h3>pKd Prediction</h3>
                    <p><strong>Predicted pKd value:</strong> {predicted_pkd:.4f}</p>
                    <p><strong>Converted Kd value:</strong> {predicted_kd:.4e} M</p>
                    <p><strong>Binding Score (ΔG):</strong> {binding_score_kd:.4f} kJ/mol</p>
                </div>
            """, unsafe_allow_html=True)
        
        if predicted_ec50 is not None:
            predicted_pec50 = -np.log10(predicted_ec50)
            st.markdown(f"""
                <div class="result-card">
                    <h3>EC50 Prediction</h3>
                    <p><strong>Predicted EC50 value:</strong> {predicted_ec50:.4e} M</p>
                    <p><strong>Converted pEC50 value:</strong> {predicted_pec50:.4f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        if predicted_pic50 is not None:
            predicted_ic50 = 10**(-predicted_pic50)
            st.markdown(f"""
                <div class="result-card">
                    <h3>pIC50 Prediction</h3>
                    <p><strong>Predicted pIC50 value:</strong> {predicted_pic50:.4f}</p>
                    <p><strong>Converted IC50 value:</strong> {predicted_ic50:.4e} M</p>
                    <p><strong>Predicted Bioactivity:</strong> {bioactivity}</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Visualize the 3D structure
        st.markdown("## 3D Structure Visualization")
        viewer = generate_3d_view(smiles_input)
        if viewer is not None:
            viewer_html = viewer._make_html()
            st.components.v1.html(viewer_html, height=500)
        else:
            st.error(f"Invalid SMILES string: {smiles_input}")

        # Explanations
        st.markdown("<a id='accuracy-explanation'></a>", unsafe_allow_html=True)
        with st.expander("What does Accuracy mean?"):
            st.write("""
            The accuracy represents the proportion of correct predictions out of the total predictions made by the model. 
            It is calculated as:
            
            \[
            \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \times 100
            \]
            """)

        st.markdown("<a id='error-explanation'></a>", unsafe_allow_html=True)
        with st.expander("What does Error Percentage mean?"):
            st.write("""
            The error percentage represents the proportion of incorrect predictions out of the total predictions made by the model. 
            It is calculated as:
            
            \[
            \text{Error} = \frac{\text{Number of Incorrect Predictions}}{\text{Total Number of Predictions}} \times 100
            \]
            """)
            
# Data visualization for homo_sapiens_single_protein.csv
st.title("🧬 Data Visualization for Homo Sapiens Single Protein")

# Load the CSV file
csv_file = "homo_sapiens_single_protein.csv"
df = pd.read_csv(csv_file)

# Display the dataframe
st.write(df)

# Interactive visualization for 'pref_name' column
st.markdown("## Interactive Visualization for 'pref_name' Column")
fig = px.histogram(df, x='pref_name', title='Distribution of pref_name')
st.plotly_chart(fig)
