import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- Configuration ---
# Create a directory to store data
DATA_DIR = 'asl_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Number of samples to collect per sign
NUM_SAMPLES = 100

# Signs to collect (static ASL alphabet, excluding J and Z)
SIGNS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We only want to track one hand for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Data Collection Function ---
def collect_data():
    """
    Cycles through each sign, capturing NUM_SAMPLES of hand landmarks for each.
    Press 'n' to move to the next sign.
    Press 'q' to quit.
    """
    
    # We will save data to a single CSV file
    csv_path = os.path.join(DATA_DIR, 'asl_landmarks.csv')
    print(f"Saving data to {csv_path}")
    
    # Create header for CSV
    # 21 landmarks, each with x, y, z. Total = 63 features
    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) # Write the header row

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

        for sign in SIGNS:
            print(f"\n--- Collecting samples for: {sign} ---")
            print(f"Get your hand ready. Press 'n' to start collecting...")
            
            sample_count = 0
            collecting = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame.")
                    break

                # Flip the frame horizontally for a natural, mirror-like view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image
                results = hands.process(image_rgb)
                
                # --- UI Text ---
                if collecting:
                    ui_text = f"Collecting {sign}: {sample_count}/{NUM_SAMPLES}"
                    color = (0, 255, 0) # Green
                else:
                    ui_text = f"Ready for {sign}? Press 'n' to start."
                    color = (0, 0, 255) # Red

                cv2.putText(frame, ui_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # --- Landmark Extraction and Drawing ---
                if results.multi_hand_landmarks:
                    # We only use the first hand detected
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    if collecting:
                        # Extract landmarks and save
                        landmarks_row = [sign] # First column is the label
                        
                        # --- Normalization (Simple Version) ---
                        # We save raw coordinates for now. 
                        # A more robust pipeline would normalize them *before* saving.
                        # For simplicity, we'll save raw and normalize in the trainer.
                        
                        all_landmarks = []
                        for lm in hand_landmarks.landmark:
                            all_landmarks.append(lm.x)
                            all_landmarks.append(lm.y)
                            all_landmarks.append(lm.z)
                        
                        # Flatten and add to our row
                        landmarks_row.extend(all_landmarks)
                        
                        # Write to CSV
                        writer.writerow(landmarks_row)
                        
                        sample_count += 1
                        if sample_count >= NUM_SAMPLES:
                            collecting = False
                            print(f"Finished collecting for {sign}.")
                            break # Move to the next sign

                # Display the frame
                cv2.imshow('Data Collection', frame)

                # --- Key Controls ---
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Quitting.")
                    return # Exit function
                
                if key == ord('n') and not collecting:
                    collecting = True
                    print("Starting collection...")
        
        cap.release()
        cv2.destroyAllWindows()
        print("--- Data collection complete! ---")

if __name__ == "__main__":
    collect_data()
