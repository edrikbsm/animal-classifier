import os
os.chdir("/Users/edrik/animal_classifier")

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Configuration ──────────────────────────────────────────
MODEL_PATH      = "models/animal_classifier.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE        = (224, 224)

# Traduction des noms italiens → français
TRADUCTIONS = {
    "cane":       "🐶 Chien",
    "gatto":      "🐱 Chat",
    "cavallo":    "🐴 Cheval",
    "farfalla":   "🦋 Papillon",
    "elefante":   "🐘 Éléphant",
    "mucca":      "🐮 Vache",
    "pecora":     "🐑 Mouton",
    "scoiattolo": "🐿️ Écureuil",
    "ragno":      "🕷️ Araignée",
    "gallina":    "🐔 Poule"
}

# ── Chargement du modèle (mis en cache pour ne pas recharger à chaque interaction) ──
@st.cache_resource
def charger_modele():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

# ── Fonction de prédiction ─────────────────────────────────
def predire(image, model, class_names):
    # Redimensionner et préprocesser
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)        # normalisation EfficientNet
    img_array = np.expand_dims(img_array, axis=0)  # ajoute la dimension batch : (1, 224, 224, 3)

    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]  # vecteur de 10 probabilités

    # Top 3
    top3_indices = np.argsort(scores)[::-1][:3]
    top3 = [(class_names[str(i)], float(scores[i])) for i in top3_indices]
    return top3

# ── Interface Streamlit ────────────────────────────────────
st.set_page_config(page_title="Classificateur d'Animaux", page_icon="🐾", layout="centered")

st.title("🐾 Classificateur d'Animaux")
st.write("Upload une photo d'animal et le modèle identifie ce que c'est !")

# Chargement du modèle
with st.spinner("Chargement du modèle..."):
    try:
        model, class_names = charger_modele()
        st.success("Modèle chargé ✅")
    except Exception as e:
        st.error(f"Modèle pas encore disponible — attends la fin de l'entraînement ! ({e})")
        st.stop()

# Upload de l'image
uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Affichage de l'image
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image uploadée", use_column_width=True)

    with col2:
        with st.spinner("Analyse en cours..."):
            top3 = predire(image, model, class_names)

        # Résultat principal
        nom_classe, score = top3[0]
        nom_fr = TRADUCTIONS.get(nom_classe, nom_classe)
        st.markdown(f"## {nom_fr}")
        st.markdown(f"**Confiance : {score*100:.1f}%**")
        st.progress(score)

        # Top 3
        st.markdown("---")
        st.markdown("**Top 3 des hypothèses :**")
        for i, (nom, sc) in enumerate(top3):
            nom_fr = TRADUCTIONS.get(nom, nom)
            st.write(f"{i+1}. {nom_fr} — {sc*100:.1f}%")
            st.progress(sc)