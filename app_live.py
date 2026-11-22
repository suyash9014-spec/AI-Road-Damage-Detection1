
import cv2
import sqlite3
import requests
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
IMAGE_DIR = BASE_DIR / "detections"
DATABASE_PATH = BASE_DIR / "road_damage.db"

WINDOW_NAME = "Road Damage Detection - YOLO11"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "D00", "D01", "D0w0", "D10", "D11",
    "D20", "D40", "D43", "D44", "D50"
]

CONFIDENCE_THRESHOLD = 0.10
SAVE_INTERVAL = 5


# ==========================================
# 2. DATABASE CREATION
# ==========================================

def create_database():

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            damage_type TEXT,
            confidence REAL,
            latitude REAL,
            longitude REAL,
            location_source TEXT,
            image_path TEXT,
            timestamp TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Database ready!")


# ==========================================
# 3. SAVE DETECTION
# ==========================================

def save_detection(
    damage_type,
    confidence,
    latitude,
    longitude,
    location_source,
    image_path
):

    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute("""
        INSERT INTO detections
        (
            damage_type,
            confidence,
            latitude,
            longitude,
            location_source,
            image_path,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        damage_type,
        confidence,
        latitude,
        longitude,
        location_source,
        str(image_path),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# ==========================================
# 4. INTERNET-BASED LOCATION
# ==========================================

def get_internet_location():

    try:

        response = requests.get(
            "https://ipapi.co/json/",
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if latitude is not None and longitude is not None:

            latitude = float(latitude)
            longitude = float(longitude)

            print("\nApproximate internet location:")
            print("Latitude:", latitude)
            print("Longitude:", longitude)

            return latitude, longitude

    except (requests.RequestException, ValueError) as error:

        print("Location service unavailable:", error)

    return None, None


# ==========================================
# 5. LOCATION SELECTION
# ==========================================

def get_location():

    print("\n========== LOCATION OPTIONS ==========")
    print("1. Internet-based approximate location")
    print("2. Enter coordinates manually")
    print("3. No location")

    choice = input("\nChoose 1, 2 or 3: ").strip()

    if choice == "1":

        latitude, longitude = get_internet_location()

        if latitude is not None:

            return latitude, longitude, "IP-based"

        print("Could not obtain internet location.")
        return None, None, "Unavailable"

    elif choice == "2":

        try:

            latitude = float(input("Enter latitude: "))
            longitude = float(input("Enter longitude: "))

            if not -90 <= latitude <= 90:
                raise ValueError("Latitude must be between -90 and 90.")

            if not -180 <= longitude <= 180:
                raise ValueError("Longitude must be between -180 and 180.")

            print("Manual location selected.")

            return latitude, longitude, "Manual"

        except ValueError as error:

            print("Invalid coordinates:", error)

            return None, None, "Unavailable"

    else:

        print("Running without location.")

        return None, None, "Unavailable"


# ==========================================
# 6. LIVE CAMERA APPLICATION
# ==========================================

def main():

    camera = None

    try:

        # Check trained model
        if not MODEL_PATH.exists():

            print("ERROR: models/best.pt not found!")
            return

        print("\nLoading YOLO11 model...")

        model = YOLO(str(MODEL_PATH))

        create_database()

        # Select location
        latitude, longitude, location_source = get_location()

        # Open webcam
        print("\nStarting laptop webcam...")

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():

            print("ERROR: Webcam could not be opened.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("\nRoad damage detection started!")
        print("Press Q or ESC inside the camera window to stop.")
        print("You can also press Ctrl+C in the terminal.")

        # Avoid saving the same class continuously
        last_saved = {}

        while True:

            success, frame = camera.read()

            if not success:

                print("Unable to read camera frame.")
                break

            # YOLO prediction
            results = model.predict(
                source=frame,
                imgsz=416,
                conf=CONFIDENCE_THRESHOLD,
                iou=0.45,
                device="cpu",
                verbose=False
            )

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            detection_count = len(results[0].boxes)

            # Display detection counter
            cv2.putText(
                annotated_frame,
                f"Detections: {detection_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                "Press Q or ESC to stop",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            current_time = time.time()

            # Save detected damage
            for box in results[0].boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id >= len(CLASS_NAMES):
                    continue

                damage_type = CLASS_NAMES[class_id]

                previous_save = last_saved.get(damage_type, 0)

                if current_time - previous_save < SAVE_INTERVAL:
                    continue

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                image_path = IMAGE_DIR / (
                    f"{damage_type}_{timestamp}.jpg"
                )

                # Save annotated camera image
                image_saved = cv2.imwrite(
                    str(image_path),
                    annotated_frame
                )

                if image_saved:

                    save_detection(
                        damage_type,
                        confidence,
                        latitude,
                        longitude,
                        location_source,
                        image_path
                    )

                    last_saved[damage_type] = current_time

                    print(
                        f"Saved: {damage_type} | "
                        f"Confidence: {confidence:.2f}"
                    )

            # Display webcam
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):

                print("\nStopping webcam...")
                break

            # Detect manually closed window
            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1:

                print("\nWebcam window closed.")
                break

    except KeyboardInterrupt:

        print("\nApplication interrupted by user.")

    except Exception as error:

        print("\nApplication error:", error)

    finally:

        # Always release webcam and close windows
        if camera is not None:

            camera.release()

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("\nCamera released.")
        print("Detection records saved in road_damage.db")
        print("Saved images are in the detections folder.")
        print("Application stopped successfully.")


# ==========================================
# 7. START APPLICATION
# ==========================================

if __name__ == "__main__":

    main()
