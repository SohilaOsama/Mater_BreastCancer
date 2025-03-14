import streamlit as st

def display_advanced_ui():
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
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧪 Molecular Property Prediction")

    st.write("""
        This app predicts various molecular properties based on the SMILES string of a molecule.
        Enter the SMILES string below to get started.
    """)