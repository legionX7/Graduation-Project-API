#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import requests
import re
from PIL import Image
from io import BytesIO
import time
from sklearn import svm
import joblib

model = joblib.load('../model.pkl')

# Replace with the IP address of your ESP32-CAM
esp32_cam_ip = "192.168.1.100"

# Get sensor values
sensor_url = f"http://{esp32_cam_ip}/"
response = requests.get(sensor_url)

if response.status_code == 200:
    sensor1_value = re.search(r"Sensor 1: (\d+)", response.text)
    sensor2_value = re.search(r"Sensor 2: (\d+)", response.text)

    if sensor1_value and sensor2_value:
        sensor1_value = sensor1_value.group(1)
        sensor2_value = sensor2_value.group(1)
        print(f"Heart_Rate: {sensor1_value}")
        print(f"TEMP : {sensor2_value}")
    else:
        print("Failed to parse sensor values")
else:
    print("Failed to retrieve sensor values")


# Load the cascade classifier
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2)
    return blurred

    # Apply morphological operations for noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    return opened


def find_pupil(frame):
    # Adjust parameters to detect smallest and darkest circle
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5, maxRadius=10)
    return circles


def estimate_pupil_size(pupil_diameter, focal_length, real_pupil_radius):
    distance_to_camera = (focal_length * real_pupil_radius) / pupil_diameter
    pixels_per_mm = pupil_diameter / real_pupil_radius
    return distance_to_camera, pixels_per_mm


def sync_photo():
    # Get photo
    photo_url = f"http://{esp32_cam_ip}/capture"
    response = requests.get(photo_url)

    if response.status_code == 200:
        photo = Image.open(BytesIO(response.content))
        return np.array(photo)
    else:
        print("Failed to retrieve photo")
        return None


def main():
    # Constants
    focal_length = 4000  # Focal length of the camera in pixels (you may need to calibrate your camera)
    real_pupil_radius = 1.25  # Approximate radius of the pupil in mm (may vary among individuals)
    last_pupil_size = None  # Initialize last pupil size variable
    while True:
        frame = sync_photo()
        if frame is not None:
            preprocessed_frame = preprocess_frame(frame)
            circles = find_pupil(preprocessed_frame)

            if circles is not None:
                # Find smallest and darkest circle
                circles = np.round(circles[0, :]).astype("int")
                gray = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR)
                smallest_circle = None
                smallest_circle_brightness = 255
                for (x, y, r) in circles:
                    mask = np.zeros(preprocessed_frame.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                    brightness = cv2.mean(preprocessed_frame, mask=mask)[0]
                    if brightness < smallest_circle_brightness:
                        smallest_circle = (x, y, r)
                        smallest_circle_brightness = brightness
                if smallest_circle is not None:
                    (x, y, r) = smallest_circle
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    distance, pixels_per_mm = estimate_pupil_size(2 * r, focal_length, real_pupil_radius)
                    # Calculate the pupil size in pixels
                    pupil_size_px = 2 * r

                    # Convert the pupil size to millimeters
                    pupil_size_mm = pupil_size_px * 25.4 / 100

                    # Print the pupil size in millimeters
                    print("pupil_diameter", pupil_size_mm)
                    last_pupil_size = pupil_size_mm

                    # cv2.putText(frame, f"Pupil diameter: {(2 * r) * 25.4 / 100:.2f} mm", (x - 50, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Pupil detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Wait for 5 seconds before syncing the photo again
        time.sleep(0.009)

    cv2.destroyAllWindows()
    # Retrieve the last detected pupil size
    if last_pupil_size is not None:
        print("Last detected Pupil diameter:", last_pupil_size)

        # Get values from algorithm or sensors
        matrix = []
        gender = input("Enter gender (male or female) : ")
        age = int(input("Enter age : "))
        pupil_r = last_pupil_size
        pupil_l = last_pupil_size
        temp = sensor2_value
        heartbeat = sensor1_value

        # Convert gender to 0 for male and 1 for female
        gender_code = 0 if gender.lower() == "male" else 1

        # Append values to the matrix
        matrix.append([gender_code, age, pupil_r, pupil_l, temp, heartbeat])
        print(matrix)

        predictions = model.predict(matrix)
        print("Predictions:", predictions)

        if predictions[0] == 0:
            print("abnormal")
        else:
            print("normal")

    if __name__ == "__main__":
        main()