from scripts import display_weights
from models import LeNet5
import streamlit as st

st.title("Afficheur de poids de modèle PyTorch")

# Widget pour uploader un fichier
uploaded_file = st.file_uploader("Upload a model file (.pth)", type=["pth"])

# Initialisation du modèle LeNet5
model = LeNet5()

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")
    display_weights(uploaded_file, model)
