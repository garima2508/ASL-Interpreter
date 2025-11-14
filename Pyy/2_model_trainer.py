import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# --- Configuration ---
DATA_FILE = 'asl_data/asl_landmarks.csv'
MODEL_SAVE_PATH = 'asl_model.keras'
SCALER_SAVE_PATH = 'asl_scaler.pkl' # To save our normalizer
CLASSES_SAVE_PATH = 'asl_classes.npy' # To save the class names
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 1. Load Data ---
print("Loading data...")
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please run '1_data_collector.py' first.")
    exit()

if data.empty:
    print("Error: Data file is empty.")
    exit()

print(f"Data shape: {data.shape}")

# Separate features (X) and labels (y)
X = data.drop('label', axis=1) # All columns except 'label'
y = data['label']

# --- 2. Preprocessing ---
print("Preprocessing data...")

# --- 2a. Normalization Strategy ---
# We make all coordinates relative to the wrist (landmark 0: x0, y0, z0)
# This makes the model robust to hand position.

# Get wrist coordinates
wrist_x = X['x0']
wrist_y = X['y0']
wrist_z = X['z0']

# Create a new DataFrame for normalized features
X_normalized = pd.DataFrame()

num_landmarks = 21
for i in range(num_landmarks):
    # Subtract wrist coordinates from all other landmarks
    # For i=0, this will just result in 0, 0, 0 which is fine.
    X_normalized[f'x{i}'] = X[f'x{i}'] - wrist_x
    X_normalized[f'y{i}'] = X[f'y{i}'] - wrist_y
    X_normalized[f'z{i}'] = X[f'z{i}'] - wrist_z


# --- 2b. Scaling ---
# We scale the data to have zero mean and unit variance
# This helps the neural network train faster and more stably
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_normalized)

# --- 2c. Save the scaler ---
# We MUST use the *same* scaler on our live data
with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {SCALER_SAVE_PATH}")

# --- 2d. Encode Labels ---
# Convert 'A', 'B', 'C' into 0, 1, 2
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder classes for later
np.save(CLASSES_SAVE_PATH, label_encoder.classes_)
print(f"Classes saved to {CLASSES_SAVE_PATH}: {label_encoder.classes_}")

# Convert 0, 1, 2 into [1,0,0], [0,1,0], [0,0,1] (one-hot encoding)
y_categorical = to_categorical(y_encoded)

num_classes = len(label_encoder.classes_)
input_shape = X_scaled.shape[1] # Should be 63

# --- 3. Split Data ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_categorical, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y_categorical # Ensure balanced classes in train/test
)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# --- 4. Build the Model ---
print("Building neural network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax') # Output layer
])

model.summary()

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Train the Model ---
print("Training model...")
EPOCHS = 75
BATCH_SIZE = 32

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# --- 7. Evaluate the Model ---
print("Evaluating model...")
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Loss: {val_loss:.4f}")

# --- 8. Save the Model ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("--- Training complete! ---")
