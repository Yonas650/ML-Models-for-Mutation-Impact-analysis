import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from imblearn.combine import SMOTEENN

#check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load Mutation Data
mutations_df = pd.read_csv('mutations.txt', sep='\t')
print("Mutation Data:")
print(mutations_df.head())

#load Protein Data
protein_df = pd.read_csv('data/brca_protein_sequences.csv')
print("\nProtein Data:")
print(protein_df.head())

#feature engineering
mutations_df['BRCA1_Status'] = mutations_df['BRCA1'].apply(lambda x: 0 if x == 'WT' else 1)
mutations_df['BRCA2_Status'] = mutations_df['BRCA2'].apply(lambda x: 0 if x == 'WT' else 1)

X = mutations_df.drop(columns=['STUDY_ID', 'SAMPLE_ID', 'BRCA1', 'BRCA2'])
y = X['BRCA1_Status']
X = X.drop(columns=['BRCA1_Status'])

#convert to numpy arrays
X = X.values
y = y.values

#define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(32 * 1, 64)
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  #add channel dimension
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  #flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return self.softmax(x)

#perform 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}/5")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    #combine SMOTE with RandomUnderSampler to avoid overcompensation
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    #convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    #create Dataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #initialize the model, loss function, and optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #train the CNN Model
    num_epochs = 50
    best_accuracy = 0
    early_stop_patience = 5
    patience_counter = 0

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

        #evaluate on the test set
        model.eval()
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_pred_list.append(predicted.cpu().numpy())
                y_true_list.append(y_batch.numpy())

        y_pred = np.concatenate(y_pred_list)
        y_true = np.concatenate(y_true_list)

        accuracy = accuracy_score(y_true, y_pred)

        #early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}')

    print(f"Best Accuracy for fold {fold + 1}: {best_accuracy:.2f}")
    fold_accuracies.append(best_accuracy)

#calculate average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy across all folds: {average_accuracy:.2f}")
