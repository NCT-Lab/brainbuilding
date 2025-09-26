import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from brainbuilding.main import ThirdOrderMomentTensorTransformer
import os

# Load the preprocessed data
DATASET_FNAME = os.getenv("DATASET_NAME", 'data/preprocessed/motor-imagery-2.npy')
data = np.load(DATASET_FNAME, allow_pickle=True)

# Filter out background samples
mask = ~data['is_background']
X_raw = data['sample'][mask]
y = data['label'][mask]
subject_ids = data['subject_id'][mask]

# Transform the data using our tensor transformer
transformer = ThirdOrderMomentTensorTransformer()
X_transformed = transformer.transform(X_raw)

# Get unique subject IDs
unique_subjects = np.unique(subject_ids)

# Initialize arrays to store metrics for each subject
precisions = []
recalls = []
f1_scores = []

# Perform LOOCV for each subject
for subject_id in unique_subjects:
    # Get data for current subject
    subject_mask = subject_ids == subject_id
    X_subject = X_transformed[subject_mask]
    y_subject = y[subject_mask]
    
    # Initialize arrays for this subject's predictions
    y_true = []
    y_pred = []
    
    # Perform LOOCV
    for i in range(len(X_subject)):
        # Split data
        X_train = np.delete(X_subject, i, axis=0)
        y_train = np.delete(y_subject, i)
        X_test = X_subject[i:i+1]
        y_test = y_subject[i:i+1]
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=25)
        knn.fit(X_train, y_train)
        
        # Predict
        pred = knn.predict(X_test)
        
        # Store results
        y_true.append(y_test[0])
        y_pred.append(pred[0])
    
    # Calculate metrics
    precisions.append(precision_score(y_true, y_pred))
    recalls.append(recall_score(y_true, y_pred))
    f1_scores.append(f1_score(y_true, y_pred))

# Create bar plot
plt.figure(figsize=(12, 6))
x = np.arange(len(unique_subjects))
width = 0.25

plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1_scores, width, label='F1 Score')

plt.xlabel('Subject ID')
plt.ylabel('Score')
plt.title('Classification Metrics per Subject (LOOCV)')
plt.xticks(x, unique_subjects)
plt.legend()

# Save plot
plt.tight_layout()
os.makedirs('scripts/3rd-order-central-moment-tensors/results', exist_ok=True)
plt.savefig('scripts/3rd-order-central-moment-tensors/results/subject_metrics.pdf')
plt.close()

# Print average metrics
print(f"Average Precision: {np.mean(precisions):.3f}")
print(f"Average Recall: {np.mean(recalls):.3f}")
print(f"Average F1 Score: {np.mean(f1_scores):.3f}") 