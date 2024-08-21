import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#load mutation data
mutations_df = pd.read_csv('mutations.txt', sep='\t')
print("Mutation Data:")
print(mutations_df.head())

#load protein data
protein_df = pd.read_csv('data/brca_protein_sequences.csv')
print("\nProtein Data:")
print(protein_df.head())

#feature engineering
mutations_df['BRCA1_Status'] = mutations_df['BRCA1'].apply(lambda x: 0 if x == 'WT' else 1)
mutations_df['BRCA2_Status'] = mutations_df['BRCA2'].apply(lambda x: 0 if x == 'WT' else 1)

X = mutations_df.drop(columns=['STUDY_ID', 'SAMPLE_ID', 'BRCA1', 'BRCA2'])
y = X['BRCA1_Status']
X = X.drop(columns=['BRCA1_Status'])

#initialize 5-fold cross-validation
kf = StratifiedKFold(n_splits=5)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"Fold {fold}/5")
    
    #split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    #apply SMOTEENN
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    
    #check the number of unique classes
    unique_classes = len(set(y_train_resampled))
    if unique_classes <= 1:
        print(f"Only one class present after resampling in fold {fold}. Skipping this fold.")
        continue

    #initialize and train the SVM model
    svm_model = make_pipeline(StandardScaler(), SVC(probability=True, class_weight='balanced', random_state=42))
    
    best_accuracy = 0
    patience = 3
    counter = 0
    
    for epoch in range(50):
        svm_model.fit(X_train_resampled, y_train_resampled)
        
        #evaluate on validation data
        y_val_pred = svm_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"Epoch [{epoch+1}/50], Accuracy: {val_accuracy:.4f}")
        
        #check for early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    print(f"Best Accuracy for fold {fold}: {best_accuracy:.2f}")
    fold_accuracies.append(best_accuracy)

#calculate average accuracy
if fold_accuracies:
    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average Accuracy across all folds: {average_accuracy:.2f}")
else:
    print("No valid folds were processed.")
