
# ML Models for Mutation Impact Analysis

This repository contains implementations of various machine learning models to predict the impact of mutations using mutation data and protein sequence data. The models implemented are:

- Random Forest
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)
- Gated Recurrent Unit (GRU)

## Data

Two main datasets are used in this project:

1. **Mutation Data**: Contains information about mutations in BRCA1 and BRCA2 genes.
2. **Protein Data**: Contains protein sequences related to BRCA1 and BRCA2 genes.

## Models and Results

### 1. Random Forest

The Random Forest model was trained using the mutation data. The model achieved the following accuracy across 5-fold cross-validation:

- **Average Accuracy**: 0.95

### 2. Support Vector Machine (SVM)

The SVM model was trained using the mutation data. The model achieved the following accuracy across 5-fold cross-validation:

- **Average Accuracy**: 0.94

### 3. Convolutional Neural Network (CNN)

The CNN model was trained using the mutation data. The model achieved the following accuracy across 5-fold cross-validation:

- **Average Accuracy**: 0.97

### 4. Gated Recurrent Unit (GRU)

The GRU model was trained using the mutation data. The model achieved the following accuracy across 5-fold cross-validation:

- **Average Accuracy**: 0.97

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-Models-for-Mutation-Impact-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML-Models-for-Mutation-Impact-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the models:
   ```bash
   python random_forest.py
   python svm.py
   python cnn.py
   python gru.py
   ```


## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- torch (for CNN and GRU models)
- matplotlib
- seaborn

## License

This project is licensed under the MIT License.
