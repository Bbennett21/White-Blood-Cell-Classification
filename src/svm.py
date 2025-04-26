import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2lab
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib.colors import ListedColormap
import os
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

# Define dataset path
dataset_path = "./bloodcells_dataset"
batch_size = 32
target_size = (256, 256)  # Target image size for processing

# Load images using TensorFlow
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path, image_size=target_size, batch_size=batch_size
)

# Get class names
class_names = dataset.class_names
print("Classes:", class_names)

# Create output directory for figures
figures_dir = "svm_figures"
os.makedirs(figures_dir, exist_ok=True)

# Preprocessing function: Resize, normalize, and extract features
def extract_features(image, label):
    # Convert image tensor to numpy array
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8).numpy()
    
    # Convert RGB to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    # Extract HOG features (shape features)
    hog_features = hog(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),  # Convert to grayscale for HOG
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        orientations=9, 
        block_norm='L2-Hys', 
        feature_vector=True
    )
    
    # Extract color features (Lab color space statistics)
    l_mean, a_mean, b_mean = np.mean(lab_image[:, :, 0]), np.mean(lab_image[:, :, 1]), np.mean(lab_image[:, :, 2])
    l_std, a_std, b_std = np.std(lab_image[:, :, 0]), np.std(lab_image[:, :, 1]), np.std(lab_image[:, :, 2])
    
    # Concatenate features
    feature_vector = np.hstack([hog_features, np.array([l_mean, a_mean, b_mean, l_std, a_std, b_std])])
    
    return feature_vector.astype(np.float32), label.numpy(), image  # Return image for visualization

# Extract features and labels from dataset
features, labels, original_images = [], [], []
for img_batch, lbl_batch in dataset:
    for img, lbl in zip(img_batch, lbl_batch):
        feat, lbl, orig_img = extract_features(img, lbl)
        features.append(feat)
        labels.append(lbl)
        original_images.append(orig_img)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)
original_images = np.array(original_images)

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2000, random_state=42)  # Adjust components based on variance explained
features_pca = pca.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    features_pca, labels, original_images, test_size=0.2, random_state=42
)

# Train SVM classifier
svm = SVC(kernel='rbf', probability=True, random_state=42)  # Added probability=True for ROC curves
svm.fit(X_train, y_train)

# Get predictions
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)

# Evaluate SVM classifier
accuracy = svm.score(X_test, y_test)
print(f"SVM Classification Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 1. Confusion Matrix
def plot_confusion_matrix():
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/confusion_matrix.png", dpi=300)
    plt.close()

# 2. ROC Curve and AUC
def plot_roc_curve():
    plt.figure(figsize=(10, 8))
    
    # For multiclass classification
    n_classes = len(class_names)
    
    # One-vs-Rest approach for ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            (y_test == i).astype(int), 
            y_pred_proba[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (class {class_names[i]}, area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/roc_curve.png", dpi=300)
    plt.close()

# 3. Precision-Recall Curve
def plot_precision_recall_curve():
    plt.figure(figsize=(10, 8))
    
    # For multiclass classification
    n_classes = len(class_names)
    
    # Plot precision-recall curve for each class
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(
            (y_test == i).astype(int), 
            y_pred_proba[:, i]
        )
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'Precision-Recall for class {class_names[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/precision_recall_curve.png", dpi=300)
    plt.close()

# 4. Decision Boundary Visualization (using first 2 PCA components)
def plot_decision_boundary():
    # Create a mesh grid for the first two PCA components
    h = 0.02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the meshgrid points using just the first 2 PCA components
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], features_pca.shape[1]-2))])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and test points
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    
    # Plot test points
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, edgecolors='k', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundaries (First 2 PCA Components)')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/decision_boundary.png", dpi=300)
    plt.close()

# 5. Feature Importance Plot
def plot_feature_importance():
    # Use permutation importance to estimate feature importance
    perm_importance = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)
    
    # Get indices of top 20 features by importance
    top_k = 20
    sorted_idx = perm_importance.importances_mean.argsort()[-top_k:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k), perm_importance.importances_mean[sorted_idx])
    plt.yticks(range(top_k), [f"PCA feature {i}" for i in sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title(f'Top {top_k} Most Important PCA Components')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/feature_importance.png", dpi=300)
    plt.close()

# 6. Sample Images with Predictions
def plot_sample_images_with_predictions():
    # Get correctly and incorrectly classified images
    correct_indices = np.where(y_pred == y_test)[0]
    incorrect_indices = np.where(y_pred != y_test)[0]
    
    # Create a figure for correctly classified images
    plt.figure(figsize=(15, 10))
    plt.suptitle('Correctly Classified Images', fontsize=16)
    
    # Plot up to 5 correctly classified images
    num_samples = min(5, len(correct_indices))
    for i in range(num_samples):
        idx = correct_indices[i]
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images_test[idx])
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig(f"{figures_dir}/correct_classifications.png", dpi=300)
    plt.close()
    
    # Create a figure for incorrectly classified images (if any)
    if len(incorrect_indices) > 0:
        plt.figure(figsize=(15, 10))
        plt.suptitle('Incorrectly Classified Images', fontsize=16)
        
        # Plot up to 5 incorrectly classified images
        num_samples = min(5, len(incorrect_indices))
        for i in range(num_samples):
            idx = incorrect_indices[i]
            plt.subplot(1, num_samples, i+1)
            plt.imshow(images_test[idx])
            plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(f"{figures_dir}/incorrect_classifications.png", dpi=300)
        plt.close()

# 7. Learning Curves
def plot_learning_curves():
    train_sizes, train_scores, test_scores = learning_curve(
        svm, features_pca, labels, cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )
    
    # Calculate mean and standard deviation
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 8))
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-validation score')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/learning_curve.png", dpi=300)
    plt.close()

# 8. PCA Variance Explained
def plot_pca_variance():
    # Calculate variance explained by each component
    pca_full = PCA().fit(features)
    plt.figure(figsize=(10, 8))
    
    # Cumulative variance
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/pca_variance.png", dpi=300)
    plt.close()

# Generate all figures
print("\nGenerating figures...")
plot_confusion_matrix()
plot_roc_curve()
plot_precision_recall_curve()
plot_decision_boundary()
plot_feature_importance()
plot_sample_images_with_predictions()
plot_learning_curves()
plot_pca_variance()

print(f"\nAll figures saved to {figures_dir}/")