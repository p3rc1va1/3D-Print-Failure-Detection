import cv2
import numpy as np
import time
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()

# Email settings from .env
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = int(os.getenv("MAIL_PORT", 587))
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SENDER_EMAIL = os.getenv("MAIL_SENDER_EMAIL")
MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "True").lower() == "true"
MAIL_USE_SSL = os.getenv("MAIL_USE_SSL", "False").lower() == "true"

# Script settings from .env
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
CAMERA_ID = os.getenv("CAMERA_ID", "Unknown Camera")
VIDEO_SOURCE_STR = os.getenv("VIDEO_SOURCE", "0")
MODEL_PATH = os.getenv("MODEL_PATH") # You might use this in load_my_model

# Convert VIDEO_SOURCE to int if it's a number, otherwise keep as string (for file paths/URLs)
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE_STR)
except ValueError:
    VIDEO_SOURCE = VIDEO_SOURCE_STR

# --- Placeholder for Model Loading and Prediction ---
# Replace these with your actual model loading and inference logic

def load_my_model(model_path=None):
    """
    Loads your pre-trained model.
    Replace this with your actual model loading code.
    """
    print(f"Attempting to load model (path if provided: {model_path})...")
    # Example:
    # if model_path and model_path.endswith(".h5"):
    #     import tensorflow as tf
    #     model = tf.keras.models.load_model(model_path)
    # elif model_path and model_path.endswith(".pt"):
    #     import torch
    #     model = torch.load(model_path)
    #     model.eval() # Set to evaluation mode
    # else:
    #     # model = your_custom_load_function(model_path)
    model = "dummy_model" # Placeholder
    print("Model loaded (or using dummy).")
    if model == "dummy_model":
        print("WARNING: Using DUMMY model. Implement actual model loading.")
    return model

def predict_failure(model, frame):
    """
    Processes a video frame with your model to detect failure.
    Replace this with your actual inference code.
    Return (True, "details") if failure is detected, (False, None) otherwise.
    """
    # Example preprocessing (if needed):
    # processed_frame = cv2.resize(frame, (224, 224))
    # processed_frame = processed_frame / 255.0 # Normalize
    # processed_frame = np.expand_dims(processed_frame, axis=0) # Add batch dimension

    # Example prediction:
    # prediction_outputs = model.predict(processed_frame) # For TensorFlow/Keras
    # with torch.no_grad(): # For PyTorch
    #    prediction_outputs = model(torch_processed_frame)
    # failure_condition = prediction_outputs[0][0] > 0.8  # Example threshold

    # --- DUMMY FAILURE DETECTION LOGIC (REMOVE/REPLACE) ---
    global frame_counter_for_dummy_failure
    frame_counter_for_dummy_failure += 1
    if frame_counter_for_dummy_failure % 300 == 0 and frame_counter_for_dummy_failure > 0: # Simulate failure
        print("DUMMY FAILURE DETECTED!")
        return True, "Simulated failure: Event triggered by frame count."
    # --- END DUMMY LOGIC ---

    # Replace with actual logic based on your model's output:
    # if actual_condition_for_failure_based_on_model_output:
    #    return True, "Specific failure type X detected by model."
    return False, None

# --- Email Sending Function ---
def send_failure_email_direct(subject: str, recipient_email: str, body_html: str):
    if not all([MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD, MAIL_SENDER_EMAIL, recipient_email]):
        print("Email configuration incomplete. Cannot send email.")
        return False

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = MAIL_SENDER_EMAIL
    msg['To'] = recipient_email

    part_html = MIMEText(body_html, 'html')
    msg.attach(part_html)

    try:
        print(f"Connecting to email server {MAIL_SERVER}:{MAIL_PORT}...")
        if MAIL_USE_SSL:
            server = smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT)
        else:
            server = smtplib.SMTP(MAIL_SERVER, MAIL_PORT)

        if MAIL_USE_TLS and not MAIL_USE_SSL: # STARTTLS is used with a non-SSL initial connection
            server.starttls()

        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# --- Main Video Processing Loop ---
if __name__ == "__main__":
    # Load your model (do this once at the start)
    model = load_my_model(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {VIDEO_SOURCE}")
        exit()

    print(f"Starting video processing from: {VIDEO_SOURCE}")
    print(f"Failure alerts will be sent to: {RECIPIENT_EMAIL}")
    print(f"Press 'q' to quit.")

    frame_counter_for_dummy_failure = 0 # For dummy detection logic
    last_alert_time = 0
    alert_cooldown_seconds = 120  # Cooldown period in seconds (e.g., 2 minutes)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            # If it's a video file, you might want to loop or just exit
            # For live streams, this might indicate a connection issue.
            time.sleep(5) # Wait a bit before retrying or exiting
            # Re-initialize VideoCapture if it's a persistent stream that dropped
            # cap.release()
            # cap = cv2.VideoCapture(VIDEO_SOURCE)
            # if not cap.isOpened():
            #     print(f"Error: Could not re-open video source: {VIDEO_SOURCE}")
            #     break
            # continue # Or break, depending on desired behavior
            break


        # --- Model Inference ---
        failure_detected, failure_details = predict_failure(model, frame.copy()) # Send a copy if predict_failure modifies it

        if failure_detected:
            current_time = time.time()
            if (current_time - last_alert_time) > alert_cooldown_seconds:
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Failure detected by model at {timestamp_str}! Details: {failure_details}")

                email_subject = f"🚨 Failure Alert: {CAMERA_ID} at {timestamp_str}"
                email_body_html = f"""
                <html>
                    <body>
                        <h1>Failure Detected!</h1>
                        <p><strong>Camera/Source ID:</strong> {CAMERA_ID}</p>
                        <p><strong>Time of Detection:</strong> {timestamp_str}</p>
                        <p><strong>Details:</strong> {failure_details or "No additional details provided by model."}</p>
                        <p>Please investigate immediately.</p>
                    </body>
                </html>
                """
                if send_failure_email_direct(email_subject, RECIPIENT_EMAIL, email_body_html):
                    last_alert_time = current_time # Update only if email was attempted/sent
                else:
                    print("Email sending failed, will retry after cooldown if failure persists.")
            else:
                print(f"Failure detected (Details: {failure_details}), but within cooldown period. No email sent.")

        # Display the resulting frame (optional)
        cv2.imshow(f'Live Video Feed - {CAMERA_ID}', frame)

        # Press 'q' to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'): # Manual trigger for testing email (simulates failure)
            print("Manual failure trigger for email test...")
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            email_subject = f"🚨 MANUAL TEST Alert: {CAMERA_ID} at {timestamp_str}"
            email_body_html = f"""
            <html><body><h1>Manual Test Failure!</h1><p>This is a test email triggered manually.</p>
            <p><strong>Camera/Source ID:</strong> {CAMERA_ID}</p>
            <p><strong>Time of Detection:</strong> {timestamp_str}</p></body></html>"""
            send_failure_email_direct(email_subject, RECIPIENT_EMAIL, email_body_html)
            last_alert_time = time.time() # Reset cooldown after manual test

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing stopped.")