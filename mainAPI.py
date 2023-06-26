import hashlib
import random
import re
import sqlite3
import time
from io import BytesIO
from typing import Optional
import cv2
import joblib
import jwt
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from fastapi import HTTPException, Header
from pydantic import BaseModel
from pydantic import EmailStr, validator

app = FastAPI(
    title="eye drug detection",
    description="API for Eye Drug Detection App. This app allows users to sign up, login, change password, and manage "
                "their user profile. It also provides integration with a voice assistant for interactive "
                "conversations.",
    version="1.0",
    docs_url='/docs',
    openapi_url='/openapi.json',
    redoc_url='/redoc',
)


@app.get("/")
async def root():
    return {
        "message": "Hello, This is a FastAPI for Eye Pupil Diameter App, add /docs to the URL to see the "
                   "documentation"
    }


# Load the machine learning model
model = joblib.load('model.pkl')


def get_sensor_values():
    sensor_url = "http://192.168.1.100/"  # Replace with the correct sensor URL
    response = requests.get(sensor_url)

    if response.status_code == 200:
        sensor_values = re.findall(r"Sensor (\d+): (\d+)", response.text)
        sensor1_value = None
        sensor2_value = None
        for sensor, value in sensor_values:
            if sensor == "1":
                sensor1_value = int(value)
            elif sensor == "2":
                sensor2_value = int(value)

        if sensor1_value is not None and sensor2_value is not None:
            return sensor1_value, sensor2_value
        else:
            raise ValueError("Failed to parse sensor values")
    else:
        raise ConnectionError("Failed to retrieve sensor values")


@app.get("/mobile")
async def predict(gender: str, age: int) -> dict:
    """
    Endpoint to predict health status based on gender, age, and sensor data.

    Args:
        gender (str): Gender of the person ('male' or 'female').
        age (int): Age of the person.

    Returns:
        dict: Result containing predicted health status and additional information.
    """
    # Convert gender to 0 or 1
    gender_value = 0 if gender.lower() == "male" else 1

    sensor1_value, sensor2_value = get_sensor_values()

    focal_length = 4000  # Focal length of the camera in pixels (you may need to calibrate your camera)
    real_pupil_radius = 1.25  # Approximate radius of the pupil in mm (may vary among individuals)

    # Open camera
    cap = cv2.VideoCapture(0)

    last_pupil_size = None  # Initialize last pupil size variable

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2)
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Find pupil
        circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5, maxRadius=10)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            smallest_circle = None
            smallest_circle_brightness = 255
            for (x, y, r) in circles:
                mask = np.zeros(opened.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                brightness = cv2.mean(opened, mask=mask)[0]
                if brightness < smallest_circle_brightness:
                    smallest_circle = (x, y, r)
                    smallest_circle_brightness = brightness
            if smallest_circle is not None:
                (x, y, r) = smallest_circle
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                distance, pixels_per_mm = (focal_length * real_pupil_radius) / (2 * r), (2 * r) / real_pupil_radius
                pupil_size_px = 2 * r
                pupil_size_mm = pupil_size_px * 25.4 / 100
                print("Pupil diameter:", pupil_size_mm)
                last_pupil_size = pupil_size_mm

        cv2.imshow("Pupil detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if last_pupil_size is not None:
        print("Last detected Pupil diameter:", last_pupil_size)

    matrix = [[gender_value, age, last_pupil_size, last_pupil_size, sensor2_value, sensor1_value]]
    predictions = model.predict(matrix)

    status = "abnormal" if predictions[0] == 0 else "normal"
    story = f" Mohammed Ali, a {age}-year-old {gender}, has been detected through eye scanning that he has received a " \
            f"dose of a drug called opioids. The eye scanning revealed that his right eye pupil is {last_pupil_size} mm " \
            f"and the left one is {last_pupil_size} mm, which is considered abnormal.\n\n Furthermore, additional " \
            f"sensor " \
            f"readings showed abnormal values for body temperature and heart rate. Mohammed Ali's body temperature is " \
            f"measured to be {sensor2_value} degrees Celsius, and his heart rate is recorded at {sensor1_value} beats " \
            f"per minute, both of which are below the normal range.\n\n Based on these findings, Mohammed Ali's " \
            f"condition is flagged as {status}, indicating potential opioid influence. It is recommended to further " \
            f"evaluate his health condition and provide necessary medical attention.\n\n Please consult a healthcare " \
            f"professional for a detailed examination and appropriate treatment."

    result = {
        "Gender": gender,
        "Age": age,
        "Last pupil size": last_pupil_size,
        "heartbeat": sensor1_value,
        "temperature": sensor2_value,
        "status": "abnormal" if predictions[0] == 0 else "normal",
        "story": story,
    }

    return result


@app.get("/hardware")
async def predict_health(gender: str, age: int):
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
            return {
                "message": "Failed to parse sensor values",
                "temp": sensor2_value,
                "heartbeat": sensor1_value
            }
    else:
        return {"message": "Failed to retrieve sensor values"}

    '''def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2, 2)
        return blurred

    def find_pupil(frame):
        # Adjust parameters to detect smallest and darkest circle
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5, maxRadius=10)
        return circles'''

    def preprocess_frame(frame):
        blurred = cv2.GaussianBlur(frame, (9, 9), 2, 2)
        return blurred

    def find_pupil(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adjust parameters to detect smallest and darkest circle
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=25)
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

                    # cv2.putText(frame, f"Pupil diameter: {(2 * r) * 25.4 / 100:.2f} mm", (x - 50, y - 30),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Pupil detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

            # Wait for 5 seconds before syncing the photo again
            time.sleep(0.003)

    cv2.destroyAllWindows()

    if last_pupil_size is not None:
        print("Last detected Pupil diameter:", last_pupil_size)

        # Get values from algorithm or sensors
        matrix = []
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
            return {
                "Gender": gender,
                "Age": age,
                "Last pupil size": last_pupil_size,
                "temp": temp,
                "heartbeat": heartbeat,
                "result": "abnormal",
            }
        else:
            return {
                "Gender": gender,
                "Age": age,
                "Last pupil size": last_pupil_size,
                "temp": temp,
                "heartbeat": heartbeat,
                "Result": "normal",

            }
    else:
        return {"message": "No pupil size detected"}


# Create SQLite database and table for users
conn = sqlite3.connect('users.db', isolation_level=None)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users '
          '(id INTEGER PRIMARY KEY, '
          'username TEXT UNIQUE, '
          'phone_number TEXT UNIQUE, '
          'email TEXT UNIQUE, '
          'password TEXT, '
          'token TEXT)')
conn.commit()

# Secret key for JWT
SECRET_KEY = "A12345678"


# Define user schema for signup
class UserSignUp(BaseModel):
    username: str
    phone_number: str
    email: EmailStr
    password: str

    @validator('phone_number')
    def validate_phone_number(cls, phone_number):
        if not phone_number:
            raise ValueError('Phone number is required')
        if len(phone_number) != 11:
            raise ValueError('Phone number should have exactly eleven digits')
        return phone_number

    @validator('email')
    def validate_email(cls, email):
        if not email:
            raise ValueError('Email is required')
        return email

    @validator('password')
    def validate_password(cls, password):
        if len(password) < 8:
            raise ValueError('Password should be at least 8 characters')
        if not any(c.isupper() for c in password):
            raise ValueError('Password should contain at least one capital letter')
        return password


# Define user schema for login
class UserLogin(BaseModel):
    email_or_phone: str
    password: str


# Define user schema for password change
class ChangePassword(BaseModel):
    current_password: str
    new_password: str


# User Profile Schema
class UserProfile(BaseModel):
    username: str
    phone_number: Optional[str]
    email: Optional[EmailStr]


@app.post("/signup")
async def signup(user: UserSignUp):
    # Check if the username already exists
    c.execute("SELECT * FROM users WHERE username=?", (user.username,))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Check if the email already exists
    c.execute("SELECT * FROM users WHERE email=?", (user.email,))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Email already exists")

    # Check if the phone number already exists
    c.execute("SELECT * FROM users WHERE phone_number=?", (user.phone_number,))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Phone number already exists")

    # Generate a random 10-digit ID for the new user
    user_id = random.randint(1000000000, 9999999999)

    # Hash the password
    hashed_password = hashlib.sha256(user.password.encode('utf-8')).hexdigest()

    # Insert the new user into the database
    c.execute("INSERT INTO users (id, username, phone_number, email, password) "
              "VALUES (?, ?, ?, ?, ?)", (user_id, user.username, user.phone_number, user.email, hashed_password))

    # Generate JWT token
    payload = {"user_id": user_id}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    # Update the token in the database
    c.execute("UPDATE users SET token=? WHERE id=?", (token, user_id))

    return {
        "message": "User created successfully",
        "user_id": user_id,
        "token": token
    }


@app.post("/login")
async def login(user: UserLogin):
    # Hash the password
    hashed_password = hashlib.sha256(user.password.encode('utf-8')).hexdigest()

    # Check if the email or phone number exists in the database
    c.execute("SELECT * FROM users WHERE email=? OR phone_number=?", (user.email_or_phone, user.email_or_phone))
    result = c.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if the password matches
    if result[4] != hashed_password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    # Generate JWT token
    payload = {"user_id": result[0]}
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    # Update the token in the database
    c.execute("UPDATE users SET token=? WHERE id=?", (token, result[0]))

    return {
        "message": "Login successful",
        "token": token
    }


@app.put("/change_password")
async def change_password(password: ChangePassword, token: str = Header(None)):
    # Decode and verify JWT token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.exceptions.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get the user ID from the token
    user_id = payload["user_id"]

    # Hash the current password
    hashed_current_password = hashlib.sha256(password.current_password.encode('utf-8')).hexdigest()

    # Check if the current password matches
    c.execute("SELECT * FROM users WHERE id=? AND password=?", (user_id, hashed_current_password))
    result = c.fetchone()
    if not result:
        raise HTTPException(status_code=401, detail="Incorrect current password")

    # Hash the new password
    hashed_new_password = hashlib.sha256(password.new_password.encode('utf-8')).hexdigest()

    # Update the password in the database
    c.execute("UPDATE users SET password=? WHERE id=?", (hashed_new_password, user_id))

    return {"message": "Password changed successfully"}


@app.get("/profile")
async def get_user_profile(token: str = Header(None)):
    # Decode and verify JWT token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.exceptions.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get the user ID from the token
    user_id = payload["user_id"]

    # Retrieve the user profile from the database
    c.execute("SELECT username, phone_number, email FROM users WHERE id=?", (user_id,))
    result = c.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "username": result[0],
        "phone_number": result[1],
        "email": result[2]
    }


@app.put("/profile")
async def update_user_profile(user_profile: UserProfile, token: str = Header(None)):
    # Decode and verify JWT token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.exceptions.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get the user ID from the token
    user_id = payload["user_id"]

    # Check if the username already exists
    c.execute("SELECT * FROM users WHERE username=? AND id != ?", (user_profile.username, user_id))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Check if the email already exists
    c.execute("SELECT * FROM users WHERE email=? AND id != ?", (user_profile.email, user_id))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Email already exists")

    # Check if the phone number already exists
    c.execute("SELECT * FROM users WHERE phone_number=? AND id != ?", (user_profile.phone_number, user_id))
    result = c.fetchone()
    if result:
        raise HTTPException(status_code=400, detail="Phone number already exists")

    # Update the user profile in the database
    c.execute("UPDATE users SET username=?, phone_number=?, email=? WHERE id=?",
              (user_profile.username, user_profile.phone_number, user_profile.email, user_id))

    return {"message": "User profile updated successfully"}


API_KEY = 'VF.DM.646388eb1419c80007bbbaa4.XHOqETFO3cvTxlgl'
VERSION_ID = '646bc'


class Message(BaseModel):
    user_id: str
    user_input: str


@app.post("/chat")
def chat(message: Message):
    url = f"https://general-runtime.voiceflow.com/state/user/{message.user_id}/interact"
    headers = {"Authorization": API_KEY}
    body = {"action": {"type": "text", "payload": message.user_input}}

    response = requests.post(url, json=body, headers=headers)
    return response.json()
