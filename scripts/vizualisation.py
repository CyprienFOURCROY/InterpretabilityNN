def display_weights(file, model):
    """
    Charge les poids dans le modèle donné et les affiche.

    Args:
        file: Objet fichier contenant les données d'un modèle PyTorch (.pth).
        model: Instance du modèle PyTorch dans lequel charger les poids.
    """
    import numpy as np
    import pandas as pd
    import torch
    import streamlit as st
    

    # Sauvegarde temporaire du fichier uploadé
    temp_path = "temp_model.pth"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    # Chargement des poids du modèle
    state_dict = torch.load(temp_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    # Affichage des poids par couche
    for name, param in model.named_parameters():
        weight_data = param.detach().numpy()
        st.write(f"Layer: {name}, Shape: {weight_data.shape}")

        if weight_data.ndim == 2:
            # Fully connected layers
            st.dataframe(pd.DataFrame(weight_data))
        elif weight_data.ndim == 1:
            # Bias vectors or similar
            st.dataframe(pd.DataFrame(weight_data.reshape(1, -1)))
        elif weight_data.ndim > 2:
            # Convolutional layers or higher-dimensional weights
            st.write("Multi-dimensional weights (e.g., for convolutional layers):")
            for i in range(min(weight_data.shape[0], 10)):  # Show at most 10 filters
                st.write(f"Filter {i + 1}/{weight_data.shape[0]}:")
                st.dataframe(
                    pd.DataFrame(weight_data[i].reshape(-1, weight_data[i].shape[-1]))
                )
