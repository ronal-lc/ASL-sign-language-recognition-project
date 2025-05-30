
"""ASL Hand Gesture Classification Model Training

This script trains a neural network model to classify American Sign Language (ASL)
hand gestures based on processed hand landmark data.

Main Steps:
1.  Loads the processed landmark data from 'data_signs.pickle'.
2.  Encodes the string labels (e.g., 'A', 'B') into numerical format.
3.  Splits the data into training and testing sets using stratified sampling
    to maintain class proportions.
4.  Defines a sequential neural network model architecture using Keras,
    including Dense layers with ReLU activation and Dropout layers for regularization.
5.  Compiles the model, specifying the Adam optimizer, sparse categorical
    crossentropy loss function (suitable for integer labels), and accuracy metric.
6.  Trains the model on the training data, validating on the test data.
7.  Evaluates the trained model on the test set to report its accuracy.
8.  Saves the trained model to 'model.keras'.

Key Dependencies:
- TensorFlow/Keras: For building and training the neural network.
- Scikit-learn: For label encoding and data splitting.
- Pickle: For loading the dataset.
"""
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset created by create_dataset.py
with open('data_signs.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['data']  # Landmark data
y = dataset['labels']  # Corresponding labels (sign names)

# Encode string labels to numerical format (e.g., 'A' -> 0, 'B' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state=42 ensures reproducibility of the split
# stratify=y_encoded ensures that the class proportions are maintained in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define the neural network model architecture
model = Sequential([
    # Input layer: Dense layer with 256 units, ReLU activation.
    # input_shape is determined by the number of features in X_train (number of landmarks).
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Dropout layer to prevent overfitting (randomly sets 30% of input units to 0)
    
    # Hidden layer 1: Dense layer with 128 units, ReLU activation.
    Dense(128, activation='relu'),
    Dropout(0.3),  # Another Dropout layer
    
    # Hidden layer 2: Dense layer with 64 units, ReLU activation.
    Dense(64, activation='relu'),
    
    # Output layer: Dense layer with units equal to the number of unique classes.
    # Softmax activation is used for multi-class classification to output probabilities for each class.
    Dense(len(set(y_train)), activation='softmax')
])

# Compile the model
# Adam optimizer is used with a learning rate of 0.0005.
# sparse_categorical_crossentropy is used as the loss function because labels are integers.
# 'accuracy' is monitored during training and evaluation.
model.compile(optimizer=Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
# epochs=30 means the model will iterate over the entire training dataset 30 times.
# validation_data=(X_test, y_test) allows monitoring of performance on the test set during training.
print("\nStarting model training...")
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
print("Model training finished.\n")

# Evaluate the model on the test set
print("Evaluating model performance...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save the trained model
model.save('model.keras')
print("Trained model saved as 'model.keras'")