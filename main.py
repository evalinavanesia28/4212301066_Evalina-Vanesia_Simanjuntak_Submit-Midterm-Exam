import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("==============================================")
print("          Letter Classification Program")
print("==============================================")
print(f"Name  : EVALINA VANESIA SIMANJUNTAK")
print(f"NIM   : 4212301066")
print(f"Class : MECHATRONICS 5C MORNING")
print("==============================================\n")

# --- Konfigurasi ---
# Sesuaikan path ini
FILE_PATH = r"C:\Users\user\Downloads\archive (5)\emnist-letters-train.csv" 

# Total sample (500 per class × 26 = 13.000)
SAMPLES_PER_CLASS = 500 
NUM_CLASSES = 26
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS

# Parameter HOG
PPC = (8, 8) 
CPB = (2, 2) 

# Parameter SVM
SVM_KERNEL = 'linear'
SVM_C = 1.0

def fix_orientation(img):
    """Fixes the EMNIST image orientation from the CSV format."""
    img_fixed = np.fliplr(img.T)
    return img_fixed


print("--- Load Dataset ---")
try:
    data_frame = pd.read_csv(FILE_PATH, header=None)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"ERROR : Filepath or File not found {FILE_PATH}")
    exit()

labels_full = None
images_flat = None


labels_full = data_frame.iloc[:, 0].values
images_flat = data_frame.iloc[:, 1:].values.astype('uint8')


images_raw = images_flat.reshape(-1, 28, 28)
images_full = np.array([fix_orientation(img) for img in images_raw])
print("Image orientation fixed.\n")


print("--- Data Sampling ---")
sampled_images = []
sampled_labels = []

print(f"Taking {SAMPLES_PER_CLASS} samples per class...")
for i in range(1, NUM_CLASSES + 1):
    class_indices = np.where(labels_full == i)[0]
    
    
    random_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
  
    
    sampled_images.append(images_full[random_indices])
    sampled_labels.append(labels_full[random_indices])

X_data = np.concatenate(sampled_images, axis=0)
y_data = np.concatenate(sampled_labels, axis=0)
print(f"Sampling complete. Total data for this experiment : {X_data.shape[0]} samples.\n")


print("--- HOG Feature Extraction ---")
hog_features = []
start_time = time.time()

print("Processing images for HOG feature extraction...")
for image in X_data:
    
    features = hog(image, pixels_per_cell=PPC, cells_per_block=CPB, visualize=False, feature_vector=True)
    hog_features.append(features)

X_features = np.array(hog_features)
end_time = time.time()
print(f"HOG extraction finished in {end_time - start_time:.2f} seconds.")
print(f"HOG feature dataset shape : {X_features.shape}\n")


print("--- Model Evaluation with LOOCV ---")


model = SVC(kernel=SVM_KERNEL, C=SVM_C)
print(f"SVM model prepared with kernel ='{SVM_KERNEL}' and C ={SVM_C}")


loo = LeaveOneOut()
print(f"Validation method: Leave-One-Out (will run {TOTAL_SAMPLES} iterations).\n")

print("===============================================================")
print("STARTING LOOCV EVALUATION.....)")
print("===============================================================")
start_cv_time = time.time()


y_pred = cross_val_predict(model, X_features, y_data, cv=loo, n_jobs=-1)


end_cv_time = time.time()
total_minutes = (end_cv_time - start_cv_time) / 60
print(f"\nLOOCV evaluation finished in {total_minutes:.2f} minutes.\n")

print("--- Display Performance Results ---")

accuracy = accuracy_score(y_data, y_pred)

print(f"Accuracy: {accuracy * 100:.4f}%\n")

print("Classification Report (Precision, Recall, F1-Score):")
report_labels = list(range(1, NUM_CLASSES + 1))
target_names = [chr(ord('A') + i - 1) for i in report_labels]


report = classification_report(y_data, y_pred, labels=report_labels, target_names=target_names, digits=4)
print(report)

print("Generating Confusion Matrix plot...")


cm = confusion_matrix(y_data, y_pred, labels=report_labels)


plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (LOOCV - HOG + SVM {SVM_KERNEL.capitalize()})\nEVALINA - 4212301066 - MK 5C PAGI', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()


plt.savefig('CONFUSION_MATRIX_LOOCV_EVALINA.png')
print("Confusion Matrix plot saved as 'CONFUSION_MATRIX_LOOCV_EVALINA.png'")
plt.show()



