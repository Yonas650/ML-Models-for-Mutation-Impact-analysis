import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Mutation Data
mutations_df = pd.read_csv('mutations.txt', sep='\t')
print("Mutation Data:")
print(mutations_df.head())

# Load Protein Data
protein_df = pd.read_csv('data/brca_protein_sequences.csv')
print("\nProtein Data:")
print(protein_df.head())

# Feature Engineering
mutations_df['BRCA1_Status'] = mutations_df['BRCA1'].apply(lambda x: 0 if x == 'WT' else 1)
mutations_df['BRCA2_Status'] = mutations_df['BRCA2'].apply(lambda x: 0 if x == 'WT' else 1)

X = mutations_df.drop(columns=['STUDY_ID', 'SAMPLE_ID', 'BRCA1', 'BRCA2'])
y = X['BRCA1_Status']
X = X.drop(columns=['BRCA1_Status'])

print(f"Columns before dropping: {list(mutations_df.columns)}")
print(f"Columns after dropping: {list(X.columns)}")

# Ensure no zero-feature dataset is used
if X.shape[1] == 0:
    raise ValueError("The feature matrix X has 0 features. Please ensure that X has the correct features before proceeding.")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store accuracy for each fold
fold_accuracies = []

# Model definition
class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=X.shape[1], hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc1(x[:, -1, :])  # Use the output of the last GRU cell
        x = self.fc2(x)
        return self.softmax(x)

# Training with Cross-Validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}/{kf.get_n_splits()}')
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Apply SMOTE to balance the training set
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_resampled.values, dtype=torch.float32).unsqueeze(2)
    y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(2)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = GRUModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

    # Training loop
    num_epochs = 50
    best_accuracy = 0.0
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Early stopping check
        model.eval()
        y_pred_list = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_pred_list.append(predicted.cpu().numpy())

        y_pred = np.concatenate(y_pred_list)
        accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            counter = 0  # reset counter if accuracy improves
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    fold_accuracies.append(best_accuracy)
    print(f"Best Accuracy for fold {fold+1}: {best_accuracy:.2f}")

# Calculate average accuracy across folds
avg_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy across all folds: {avg_accuracy:.2f}")

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Plot and Save Confusion Matrix for the last fold
cm = confusion_matrix(y_test_tensor.cpu().numpy(), y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['WT', 'Mutant'], yticklabels=['WT', 'Mutant'])
plt.title('Confusion Matrix - GRU')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('visualizations/gru_confusion_matrix.png')
plt.close()
