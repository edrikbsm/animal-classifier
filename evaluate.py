import os
os.chdir("/Users/edrik/animal_classifier")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

# ── Configuration ──────────────────────────────────────────
DATA_DIR    = "data/raw-img"
MODEL_PATH  = "models/animal_classifier.h5"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# ── Chargement du modèle ───────────────────────────────────
print("Chargement du modèle...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Chargement des données de validation ───────────────────
datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # important : ne pas mélanger pour comparer avec les vraies classes
)

# ── Prédictions sur tout le jeu de validation ──────────────
print("Calcul des prédictions...")
predictions = model.predict(val_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)  # indice de la classe prédite
y_true = val_data.classes                # vraies classes

# ── Noms des classes ───────────────────────────────────────
class_indices = val_data.class_indices
traductions = {
    "cane":       "Chien",
    "gatto":      "Chat",
    "cavallo":    "Cheval",
    "farfalla":   "Papillon",
    "elefante":   "Elephant",
    "mucca":      "Vache",
    "pecora":     "Mouton",
    "scoiattolo": "Ecureuil",
    "ragno":      "Araignee",
    "gallina":    "Poule"
}

class_names = [traductions[name] for name in class_indices.keys()]

# ── Rapport de classification ──────────────────────────────
print("\n── Rapport de classification ──")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Matrice de confusion ───────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,          # affiche les chiffres dans chaque case
    fmt="d",             # format entier
    cmap="Blues",        # palette de couleurs
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Matrice de Confusion", fontsize=14, fontweight="bold")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("models/matrice_confusion.png", dpi=150)
plt.show()
print("\nMatrice sauvegardée dans models/matrice_confusion.png")