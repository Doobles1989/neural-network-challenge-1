# Student Loan Risk with Deep Learning

This project implements a neural network to predict the risk level of student loans using a dataset containing various loan-related features. The model is built with TensorFlow's Keras library and trained on preprocessed data to classify loans into different risk categories.

---

## Project Overview

### Goals
- Predict loan risk levels based on input features.
- Build, train, and evaluate a deep learning model for classification tasks.

---

## Workflow

### 1. Data Preparation
- **Dataset**: 
  The dataset `student-loans.csv` is loaded and explored for relevant features and target variables.
- **Features**: 
  Columns such as `payment_history`, `location_parameter`, `stem_degree_score`, etc., are used as inputs.
- **Target Variable**: 
  The `credit_ranking` column serves as the target, representing loan risk categories.

```python
# Load dataset
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
loans_df = pd.read_csv(file_path)

# Define target and features
y = loans_df["credit_ranking"]
X = loans_df.drop("credit_ranking", axis=1)

2. Data Preprocessing
	•	Train-Test Split:
The dataset is split into training (75%) and testing (25%) subsets using scikit-learn.
	•	Feature Scaling:
Numerical features are scaled using StandardScaler for better model performance.

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

3. Neural Network Model
	•	Architecture:
	•	Input layer: Matches the number of features in the dataset.
	•	Hidden layers: Two dense layers with 8 and 5 nodes respectively, using the relu activation function.
	•	Output layer: Three neurons for multi-class classification using softmax.
	•	Compilation:
	•	Loss function: sparse_categorical_crossentropy (for multi-class classification).
	•	Optimizer: adam (adaptive learning rate optimization).
	•	Metric: Accuracy.

# Create the Sequential model
nn = Sequential()

# Add layers
nn.add(Dense(units=8, input_dim=len(X_train.columns), activation="relu"))
nn.add(Dense(units=5, activation="relu"))
nn.add(Dense(units=3, activation="softmax"))

# Compile the model
nn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

4. Training and Evaluation
	•	Training:
The model is trained on the scaled training data for 50 epochs.
	•	Evaluation:
Loss and accuracy are calculated on the test dataset to evaluate model performance.

# Train the model
model = nn.fit(X_train_scaled, y_train, epochs=50)

# Evaluate the model
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model Loss: {model_loss}, Model Accuracy: {model_accuracy}")

Results
	•	Accuracy: The model achieved an accuracy of approximately 76% on the test dataset.
	•	Loss: The model’s loss value indicates the error in predictions, showing room for optimization.

Project Files
	•	student-loans.csv: Dataset containing features and target variables for loan classification.
	•	deep_learning_student_loans.py: Python script with complete code for data preparation, model training, and evaluation.

Requirements
	•	Python: 3.7 or higher.
	•	Libraries:
	•	pandas
	•	tensorflow
	•	scikit-learn

To install the required libraries, run:

pip install pandas tensorflow scikit-learn

Running the Project
	1.	Clone the repository.
	2.	Ensure all dependencies are installed.
	3.	Run the script:

python deep_learning_student_loans.py


	4.	View the performance metrics in the terminal.

Future Work
	•	Model Improvements:
	•	Add more hidden layers or adjust the number of nodes.
	•	Experiment with different activation functions.
	•	Optimize hyperparameters (e.g., learning rate, batch size).
	•	Dataset Expansion:
	•	Incorporate more features and data points for robust training.
	•	Advanced Techniques:
	•	Implement dropout layers to prevent overfitting.
	•	Use cross-validation for model evaluation.

Acknowledgements
	•	TensorFlow Keras Documentation
	•	scikit-learn Documentation
