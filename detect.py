#importing libraries
import cv2
import numpy as np
import time
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# --- Configuration Loading ---
load_dotenv()

# Email settings from .env
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = os.getenv("MAIL_PORT")
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SENDER_EMAIL = os.getenv("MAIL_SENDER_EMAIL")
MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "True").lower() == "true"
MAIL_USE_SSL = os.getenv("MAIL_USE_SSL", "False").lower() == "true"



#.env
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
CAMERA_ID = os.getenv("CAMERA_ID", "Unknown Camera")
VIDEO_SOURCE_STR = os.getenv("VIDEO_SOURCE", "0")
MODEL_PATH = os.getenv("MODEL_PATH") 
PRINTER_IP='http://192.168.34.111:8080/?action=stream'

def load_my_model():
    """
    Loads your pre-trained model.
    """
    model_path = 'content/runs/detect/train2/weights/best.pt'
    print(f"Attempting to load model (path if provided: {model_path})...")
    model = YOLO(model_path)
    return model

def predict_failure(model, frame):
    """
    Processes a video frame with a trained YOLO model to detect specified failure conditions.

    """
    results = model.predict(source=frame, imgsz=640, verbose=False, conf=0.7)

    if not results or not results[0].boxes:
        return False, []  # No detections or an issue with prediction

    detections = results[0]
    class_names_map = detections.names

    detected_failures_info = []

    for box_data in detections.boxes:
        class_id = int(box_data.cls.item())
        confidence = box_data.conf.item()
        class_name = class_names_map.get(class_id, "Unknown")

        if class_name == 'fail':
            # Bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
            detail_str = f"Failure: Detected '{class_name}' (conf: {confidence:.2f}) at [{x1},{y1},{x2},{y2}]"
            
            detected_failures_info.append({
                'class_name': class_name,
                'confidence': confidence,
                'box': [x1, y1, x2, y2],
                'detail_string': detail_str
            })

    if detected_failures_info:
        return True, detected_failures_info
    else:
        return False, []

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

#---------------------------------------------
#stop the printer function //the api call doesnt work
def pause_printer():
    options = webdriver.ChromeOptions()

    driver = webdriver.Chrome(options=options)
    driver.get("http://192.168.34.111")
    time.sleep(5)

    try:
        pause_button = driver.find_element(By.XPATH, '//button[span[contains(text(), "Pause")]]')
        pause_button.click()
        print(" Pause button clicked via XPath!")

    except Exception as e1:
        print(" XPath method failed:", e1)
        try:
            buttons = driver.find_elements(By.CLASS_NAME, 'el-button--small')
            clicked = False
            for b in buttons:
                print("Found button with text:", b.text)
                if "Pause" in b.text:
                    b.click()
                    print(" Pause button clicked via class match!")
                    clicked = True
                    break
            if not clicked:
                print(" Could not find Pause button by class match either.")
        except Exception as e2:
            print(" Class match method failed:", e2)
    finally:
        time.sleep(2)
        driver.quit()
        print(" Browser closed")
#---------------------------------------------------------


# --- Main Video Processing Loop ---
if __name__ == "__main__":
    # Load your model (do this once at the start)
    model = load_my_model()

    cap = cv2.VideoCapture('http://192.168.34.111:8080/?action=stream')
    if not cap.isOpened():
        print(f"Error: Could not open video source:")
        exit()

    print(f"Starting video processing")
    print(f"Failure alerts will be sent to: {RECIPIENT_EMAIL}")
    print(f"Press 'q' to quit.")

    last_alert_time = 0
    alert_cooldown_seconds = 120  # Cooldown period in seconds (e.g., 2 minutes)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end or camera disconnected?). Exiting ...")
            break

        # --- Model Inference ---
        # Pass the original frame to predict_failure
        failure_detected, detected_failures_info = predict_failure(model, frame)
        
        aggregated_failure_details_for_email = ""

        if failure_detected:
            current_time = time.time()
            
            # Aggregate details for email and draw boxes
            detail_strings_for_email = []
            for failure_info in detected_failures_info:
                detail_strings_for_email.append(failure_info['detail_string'])
                
                # --- Drawing on the frame ---
                x1, y1, x2, y2 = failure_info['box']
                label = f"{failure_info['class_name']}: {failure_info['confidence']:.2f}"

                # Draw rectangle (BGR color: Red)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Put label
                # Calculate text size to draw a background for better visibility
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1 - 5), (0, 0, 255), -1) # Filled background
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # White text

            aggregated_failure_details_for_email = "; ".join(detail_strings_for_email)

            if (current_time - last_alert_time) > alert_cooldown_seconds:
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Failure detected by model at {timestamp_str}! Details: {aggregated_failure_details_for_email}")

                # Send Email
                email_subject = f"🚨 Failure Alert: {CAMERA_ID} at {timestamp_str}"
                email_body_html = f"""
                <html>
                    <body>
                        <h1>Failure Detected!</h1>
                        <p><strong>Camera/Source ID:</strong> {CAMERA_ID}</p>
                        <p><strong>Time of Detection:</strong> {timestamp_str}</p>
                        <p><strong>Details:</strong> {aggregated_failure_details_for_email or "No additional details provided by model."}</p>
                        <p>Please investigate immediately.</p>
                    </body>
                </html>
                """
                if RECIPIENT_EMAIL: # Only send if RECIPIENT_EMAIL is set
                    if send_failure_email_direct(email_subject, RECIPIENT_EMAIL, email_body_html):
                        last_alert_time = current_time
                    else:
                        print("Email sending failed, will retry after cooldown if failure persists.")
                else:
                    print("RECIPIENT_EMAIL not set. Skipping email notification.")
                    last_alert_time = current_time # Still update cooldown to avoid spamming console

                # Stop Printer
                if PRINTER_IP:
                    print(f"Attempting to stop printer at IP: {PRINTER_IP}")
                    pause_printer()
                else:
                    print("PRINTER_IP not set in .env file. Cannot stop printer.")

            else:
                print(f"Failure detected (Details: {aggregated_failure_details_for_email}), but within cooldown period. No new alert/action.")

        # Display the resulting frame
        cv2.imshow(f'Live Video Feed - {CAMERA_ID}', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('f'): 
            print("Manual failure trigger for email test...")
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            email_subject = f"🚨 MANUAL TEST Alert: {CAMERA_ID} at {timestamp_str}"
            email_body_html = f"""
            <html><body><h1>Manual Test Failure!</h1><p>This is a test email triggered manually.</p>
            <p><strong>Camera/Source ID:</strong> {CAMERA_ID}</p>
            <p><strong>Time of Detection:</strong> {timestamp_str}</p></body></html>"""
            if RECIPIENT_EMAIL:
                send_failure_email_direct(email_subject, RECIPIENT_EMAIL, email_body_html)
            else:
                print("RECIPIENT_EMAIL not set. Cannot send manual test email.")
            last_alert_time = time.time() 

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing stopped.")