import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# --- Configuration ---
MODEL_PATH = 'asl_model.keras'
SCALER_PATH = 'asl_scaler.pkl'
CLASSES_PATH = 'asl_classes.npy'

# --- Load Model and Preprocessors ---
print("Loading model and preprocessors...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # --- THIS IS THE FIX ---
    # We add allow_pickle=True to load the object array
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    # --- END OF FIX ---
    
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please run '2_model_trainer.py' first.")
    exit()

print("Model and preprocessors loaded successfully.")
print(f"Classes: {classes}")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Smoothing Buffer ---
PREDICTION_BUFFER_SIZE = 10
prediction_buffer = []
current_prediction = ""

# --- Real-Time Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(image_rgb)
    
    # Reset prediction if no hand is detected
    predicted_letter = ""
    
    if results.multi_hand_landmarks:
        # Get the first (and only) hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # --- 1. Extract Landmarks ---
        raw_landmarks = []
        for lm in hand_landmarks.landmark:
            raw_landmarks.extend([lm.x, lm.y, lm.z])
        
        # --- 2. Apply Normalization (SAME AS TRAINING) ---
        
        # Convert to numpy array
        raw_landmarks_np = np.array(raw_landmarks).reshape(1, -1) # Shape (1, 63)
        
        # Create a DataFrame-like structure for easy subtraction
        # (This is inefficient but mimics the training logic exactly)
        wrist_x = raw_landmarks_np[0, 0]
        wrist_y = raw_landmarks_np[0, 1]
        wrist_z = raw_landmarks_np[0, 2]
        
        normalized_landmarks = []
        for i in range(21):
            idx = i * 3
            normalized_landmarks.append(raw_landmarks_np[0, idx + 0] - wrist_x)
            normalized_landmarks.append(raw_landmarks_np[0, idx + 1] - wrist_y)
            normalized_landmarks.append(raw_landmarks_np[0, idx + 2] - wrist_z)
            
        normalized_landmarks_np = np.array(normalized_landmarks).reshape(1, -1)

        # --- 3. Apply Scaling (SAME AS TRAINING) ---
        scaled_landmarks = scaler.transform(normalized_landmarks_np)
        
        # --- 4. Predict ---
        prediction = model.predict(scaled_landmarks, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence > 0.7: # Confidence threshold
            predicted_letter = classes[predicted_index]
        else:
            predicted_letter = ""

        # --- Draw landmarks ---
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # --- Smoothing Logic ---
    prediction_buffer.append(predicted_letter)
    if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
        prediction_buffer.pop(0) # Remove oldest prediction

    # Check if the most common element is dominant
    if len(prediction_buffer) > 0:
        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
        # Only update if the most common prediction is consistent
        if prediction_buffer.count(most_common) > int(PREDICTION_BUFFER_SIZE * 0.6): # 60% stable
             current_prediction = most_common
   
    # --- Display Prediction ---
    # Draw a semi-transparent rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (150, 120), (0, 0, 0), -1)
    alpha = 0.6 # Transparency
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Put text
    cv2.putText(frame, "PREDICTION:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, current_prediction, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('ASL Interpreter', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

