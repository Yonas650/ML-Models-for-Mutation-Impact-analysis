import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN

#load Mutation Data
mutations_df = pd.read_csv('mutations.txt', sep='\t')
print("Mutation Data:")
print(mutations_df.head())

#load Protein Data
protein_df = pd.read_csv('data/brca_protein_sequences.csv')
print("\nProtein Data:")
print(protein_df.head())

#feature Engineering
mutations_df['BRCA1_Status'] = mutations_df['BRCA1'].apply(lambda x: 0 if x == 'WT' else 1)
mutations_df['BRCA2_Status'] = mutations_df['BRCA2'].apply(lambda x: 0 if x == 'WT' else 1)

X = mutations_df.drop(columns=['STUDY_ID', 'SAMPLE_ID', 'BRCA1', 'BRCA2'])
y = X['BRCA1_Status']
X = X.drop(columns=['BRCA1_Status'])

#5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print(f'Fold {fold}/5')

    #split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #resampling
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    #skip fold if only one class present after resampling
    if len(set(y_train_resampled)) < 2:
        print(f"Only one class present after resampling in fold {fold}. Skipping this fold.")
        continue

    #initialize the RandomForest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )

    #train the model
    rf_model.fit(X_train_resampled, y_train_resampled)

    #evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Accuracy for fold {fold}: {accuracy:.2f}")
    fold_accuracies.append(accuracy)

#average accuracy across all folds
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print(f'Average Accuracy across all folds: {average_accuracy:.2f}')

