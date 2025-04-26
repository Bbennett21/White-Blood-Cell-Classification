# Standard library imports
import os
from collections import Counter

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

# Deep learning and image processing
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



# Configuration
DATA_DIR = "./bloodcells_dataset"
IMAGE_SIZE = (224, 224)
FEATURE_FILE = "resnet50_features.npy"
LABEL_FILE = "resnet50_labels.npy"

# Load RESNET50 Model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Extract Features 
features = []
labels = []

print("📦 Starting feature extraction...")
for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    print(f"🔍 Processing class: {class_name}")
    for fname in tqdm(os.listdir(class_dir)):
        if not fname.lower().endswith(".jpg"):
            continue

        try:
            img_path = os.path.join(class_dir, fname)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            feature = feature_extractor.predict(img_array, verbose=0)
            features.append(feature.flatten())
            labels.append(class_name)
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

X = np.array(features)
y = np.array(labels)

print("\n✅ Feature extraction complete!")


#Split Dataset: 80% Train, 10% Val, 10% Test 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Dataset split complete:")
print(f"  Training:   {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Testing:    {X_test.shape[0]} samples")

# Train Random Forest

clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate

def evaluate_model(model, X, y, name="Dataset"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n Evaluation on {name}:")
    print(f"  Accuracy: {acc:.4f}")
    print("  Classification Report:")
    print(classification_report(y, y_pred))

# Evaluate on validation and test sets
evaluate_model(clf, X_val, y_val, "Validation Set")
evaluate_model(clf, X_test, y_test, "Test Set")

# Confusion Matrix Plot

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Plot confusion matrix for test set
y_test_pred = clf.predict(X_test)
unique_classes = sorted(np.unique(y))  # or use list(class_counts.keys())
plot_confusion_matrix(y_test, y_test_pred, unique_classes)

# Simplified Tree (Depth 2)
tree_index = 0
tree = clf.estimators_[tree_index]
feature_names = [f"f{i}" for i in range(X.shape[1])]

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    filled=True,
    max_depth=2, 
    feature_names=feature_names,
    class_names=unique_classes,
    rounded=True,
    fontsize=10
)
plt.title(f"Simplified Decision Tree #{tree_index} (Depth 2)")
plt.tight_layout(pad=1.0)
plt.savefig("tree_depth_2.png", dpi=300, bbox_inches='tight')
plt.show()

# Decision Tree (Depth 3)
plt.figure(figsize=(25, 15))
ax = plt.gca()

plot_tree(
    tree,
    filled=True,
    max_depth=3,
    feature_names=feature_names,
    class_names=unique_classes,
    rounded=True,
    fontsize=9,
    precision=3,
    ax=ax
)

# Increase spacing between nodes and simplify text
for text in ax.texts:
    text.set_size(9) 
    # Modify text to make it more compact
    content = text.get_text()
    # Simplify the "value" arrays by shortening to first few and last few elements
    if "value" in content and len(content) > 100:
        parts = content.split("\n")
        for i, part in enumerate(parts):
            if part.startswith("value = ["):
                # Extract just a few elements from the beginning and end
                elements = part[part.find("[")+1:part.find("]")].split(",")
                if len(elements) > 6:
                    shortened = "[" + ", ".join(elements[:3]) + ", ..., " + ", ".join(elements[-3:]) + "]"
                    parts[i] = "value = " + shortened
        content = "\n".join(parts)
        text.set_text(content)

plt.title(f"Decision Tree #{tree_index} (Depth 3)")
plt.tight_layout(pad=1.2)
plt.savefig("tree_depth_3.png", dpi=300, bbox_inches='tight')
plt.show()

# Forest Feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = min(20, len(importances))

plt.figure(figsize=(12, 8))
plt.bar(range(top_n), importances[indices][:top_n], color='skyblue')
plt.xticks(range(top_n), [f"f{indices[i]}" for i in range(top_n)], rotation=45)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Top 20 Features by Importance in Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_.png", dpi=300)
plt.show()