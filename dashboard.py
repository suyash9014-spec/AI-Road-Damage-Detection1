import sqlite3

import threading

import time

from datetime import datetime

from pathlib import Path



import av

import cv2

import folium

import pandas as pd

import requests

import streamlit as st

from streamlit_folium import st_folium

from streamlit_js_eval import get_geolocation

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from ultralytics import YOLO





# =========================================================

# PAGE CONFIGURATION

# =========================================================



st.set_page_config(

    page_title="Road Damage AI",

    page_icon="🚧",

    layout="wide",

    initial_sidebar_state="expanded",

)



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

DATABASE_PATH = BASE_DIR / "road_damage.db"

IMAGE_DIR = BASE_DIR / "detections"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)



CLASS_NAMES = [

    "D00", "D01", "D0w0", "D10", "D11",

    "D20", "D40", "D43", "D44", "D50",

]





# =========================================================

# WEBSITE DESIGN - DESKTOP AND MOBILE

# =========================================================



st.markdown(

    """

    <style>

    .stApp {

        background-color: #0e1117;

        color: #f8fafc;

    }



    .block-container {

        padding-top: 1.5rem;

        padding-bottom: 2rem;

        max-width: 1500px;

    }



    .hero {

        padding: 25px;

        border-radius: 15px;

        background: linear-gradient(120deg, #102a43, #176b87);

        color: white;

        margin-bottom: 25px;

    }



    .hero h1 {

        color: #ffffff !important;

        font-size: 34px;

    }



    .hero p {

        color: #e0f2fe !important;

        font-size: 16px;

    }



    [data-testid="stSidebar"] {

        background-color: #172033 !important;

    }



    [data-testid="stSidebar"] * {

        color: #f8fafc !important;

    }



    [data-testid="stMetric"] {

        background-color: #ffffff !important;

        padding: 20px !important;

        border-radius: 14px !important;

        border: 1px solid #e5e7eb !important;

    }



    [data-testid="stMetricLabel"],

    [data-testid="stMetricLabel"] p {

        color: #374151 !important;

    }



    [data-testid="stMetricValue"],

    [data-testid="stMetricValue"] div {

        color: #111827 !important;

    }



    .stApp h1, .stApp h2, .stApp h3 {

        color: #f8fafc !important;

    }



    .stApp p,

    .stApp label,

    .stApp [data-testid="stWidgetLabel"] {

        color: #e5e7eb;

    }



    .stApp input,

    .stApp textarea {

        color: #111827 !important;

    }



    .stApp [data-baseweb="select"] {

        color: #111827 !important;

    }



    [data-testid="stDataFrame"] {

        background-color: #ffffff;

    }



    .stButton button,

    .stDownloadButton button {

        border-radius: 9px;

        font-weight: 600;

    }



    [data-testid="stAlert"] {

        border-radius: 12px;

    }



    @media (max-width: 768px) {

        .block-container {

            padding-left: 1rem !important;

            padding-right: 1rem !important;

            padding-top: 1rem !important;

        }



        .hero {

            padding: 18px;

        }



        .hero h1 {

            font-size: 24px !important;

        }



        .hero p {

            font-size: 14px !important;

        }



        [data-testid="stMetric"] {

            padding: 13px !important;

            margin-bottom: 8px !important;

        }



        [data-testid="stMetricValue"] {

            font-size: 25px !important;

        }



        [data-testid="stMetricLabel"] {

            font-size: 13px !important;

        }



        .stButton button {

            width: 100%;

        }

    }

    </style>

    """,

    unsafe_allow_html=True,

)





# =========================================================

# DATABASE FUNCTIONS

# =========================================================



def create_database():

    """Create the detections table and add missing columns safely."""

    with sqlite3.connect(DATABASE_PATH) as conn:

        conn.execute(

            """

            CREATE TABLE IF NOT EXISTS detections (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                damage_type TEXT,

                confidence REAL,

                latitude REAL,

                longitude REAL,

                location_source TEXT,

                image_path TEXT,

                timestamp TEXT,

                gps_accuracy REAL

            )

            """

        )



        columns = [

            row[1]

            for row in conn.execute(

                "PRAGMA table_info(detections)"

            ).fetchall()

        ]



        if "gps_accuracy" not in columns:

            conn.execute(

                "ALTER TABLE detections ADD COLUMN gps_accuracy REAL"

            )



        if "location_source" not in columns:

            conn.execute(

                "ALTER TABLE detections ADD COLUMN location_source TEXT"

            )





def load_data():

    create_database()

    with sqlite3.connect(DATABASE_PATH) as conn:

        return pd.read_sql_query(

            "SELECT * FROM detections ORDER BY id DESC",

            conn,

        )





def delete_all_detection_images():

    """Delete only files inside the detections folder and clear DB rows."""

    create_database()

    deleted_count = 0

    safe_image_dir = IMAGE_DIR.resolve()



    with sqlite3.connect(DATABASE_PATH) as conn:

        records = conn.execute(

            "SELECT image_path FROM detections"

        ).fetchall()



        for (image_path,) in records:

            if not image_path:

                continue



            try:

                path = Path(image_path).resolve()

                if (

                    path.is_relative_to(safe_image_dir)

                    and path.is_file()

                ):

                    path.unlink()

                    deleted_count += 1

            except (OSError, ValueError) as error:

                print("Image deletion error:", error)



        conn.execute("DELETE FROM detections")



    return deleted_count





# =========================================================

# MODEL AND LOCATION FUNCTIONS

# =========================================================



@st.cache_resource

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model file not found: {MODEL_PATH}\n"

            "Make sure models/best.pt exists."

        )

    return YOLO(str(MODEL_PATH))





def get_ip_location():

    """Get an approximate location based on the internet IP address."""

    try:

        response = requests.get(

            "https://ipapi.co/json/",

            timeout=5,

        )

        response.raise_for_status()

        data = response.json()

        return float(data["latitude"]), float(data["longitude"])

    except Exception as error:

        print("IP location error:", error)

        return None





# =========================================================

# LIVE CAMERA PROCESSOR

# =========================================================



class RoadDamageProcessor(VideoProcessorBase):

    def __init__(

        self,

        model,

        latitude,

        longitude,

        location_source,

        gps_accuracy,

        confidence,

    ):

        self.model = model

        self.latitude = latitude

        self.longitude = longitude

        self.location_source = location_source

        self.gps_accuracy = gps_accuracy

        self.confidence = confidence

        self.last_saved = {}

        self.lock = threading.Lock()



    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")



        try:

            results = self.model.predict(

                source=image,

                imgsz=416,

                conf=self.confidence,

                device="cpu",

                verbose=False,

            )



            result = results[0]

            annotated = result.plot()

            boxes = result.boxes

            detection_count = 0



            if boxes is not None:

                for box in boxes:

                    class_id = int(box.cls[0])

                    score = float(box.conf[0])



                    if class_id < 0 or class_id >= len(CLASS_NAMES):

                        continue



                    damage_name = CLASS_NAMES[class_id]

                    detection_count += 1

                    current_time = time.time()



                    # Save each damage category at most once every 5 seconds.

                    with self.lock:

                        previous_time = self.last_saved.get(

                            damage_name, 0

                        )

                        if current_time - previous_time < 5:

                            continue

                        self.last_saved[damage_name] = current_time



                    timestamp_file = datetime.now().strftime(

                        "%Y%m%d_%H%M%S"

                    )

                    image_name = (

                        f"{damage_name}_{timestamp_file}_"

                        f"{time.time_ns()}.jpg"

                    )

                    image_path = IMAGE_DIR / image_name



                    cv2.imwrite(str(image_path), annotated)



                    with sqlite3.connect(

                        DATABASE_PATH,

                        timeout=10,

                    ) as conn:

                        conn.execute(

                            """

                            INSERT INTO detections (

                                damage_type,

                                confidence,

                                latitude,

                                longitude,

                                location_source,

                                image_path,

                                timestamp,

                                gps_accuracy

                            )

                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

                            """,

                            (

                                damage_name,

                                score,

                                self.latitude,

                                self.longitude,

                                self.location_source,

                                str(image_path),

                                datetime.now().strftime(

                                    "%Y-%m-%d %H:%M:%S"

                                ),

                                self.gps_accuracy,

                            ),

                        )



            cv2.putText(

                annotated,

                f"Damage detected: {detection_count}",

                (15, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2,

            )



            return av.VideoFrame.from_ndarray(

                annotated,

                format="bgr24",

            )



        except Exception as error:

            print("Camera processing error:", error)

            return av.VideoFrame.from_ndarray(

                image,

                format="bgr24",

            )





# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🚧 AI Road Damage Detection System</h1>
        <p>
            Intelligent road monitoring, damage detection,
            geographical mapping and analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



# =========================================================

# SIDEBAR NAVIGATION AND SETTINGS

# =========================================================



st.sidebar.title("⚙️ System Settings")



page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📷 Live Camera",

        "🗺️ Damage Map",

        "📊 Analytics",

        "🖼️ Detection Gallery",

    ],

)



st.sidebar.divider()

st.sidebar.subheader("AI Configuration")



confidence = st.sidebar.slider(

    "Detection Confidence",

    min_value=0.05,

    max_value=0.90,

    value=0.10,

    step=0.05,

)



st.sidebar.caption(

    "Lower confidence can detect more possible damage "

    "but may increase false detections."

)





# =========================================================

# LOAD DATABASE

# =========================================================



create_database()

df = load_data()



if not df.empty:

    for column in [

        "confidence",

        "latitude",

        "longitude",

        "gps_accuracy",

    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(

                df[column],

                errors="coerce",

            )



    df["damage_type"] = df["damage_type"].fillna("Unknown")





# =========================================================

# DASHBOARD PAGE

# =========================================================



if page == "🏠 Dashboard":

    st.subheader("📊 System Overview")



    total = len(df)

    mapped = 0



    if not df.empty:

        mapped = int(

            (

                df["latitude"].notna()

                & df["longitude"].notna()

            ).sum()

        )



    images = 0

    if not df.empty:

        images = sum(

            Path(path).is_file()

            for path in df["image_path"].fillna("")

            if path

        )



    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Detections", total)

    c2.metric("Mapped Records", mapped)

    c3.metric(

        "Damage Categories",

        0 if df.empty else df["damage_type"].nunique(),

    )

    c4.metric("Saved Images", images)



    st.divider()



    if df.empty:

        st.info(

            "No detections recorded yet. "

            "Open Live Camera to start detecting road damage."

        )

    else:

        left, right = st.columns(2)



        with left:

            st.subheader("Damage Category Distribution")

            counts = df["damage_type"].value_counts()

            st.bar_chart(counts)



        with right:

            st.subheader("Recent Detection Records")



            recent_records = df[

                [

                    "damage_type",

                    "confidence",

                    "latitude",

                    "longitude",

                    "timestamp",

                ]

            ].head(10).copy()



            # Format GPS values for readability.

            recent_records["latitude"] = recent_records[

                "latitude"

            ].apply(

                lambda value: (

                    "Not captured"

                    if pd.isna(value)

                    else f"{value:.6f}"

                )

            )



            recent_records["longitude"] = recent_records[

                "longitude"

            ].apply(

                lambda value: (

                    "Not captured"

                    if pd.isna(value)

                    else f"{value:.6f}"

                )

            )



            recent_records = recent_records.rename(

                columns={

                    "damage_type": "Damage Type",

                    "confidence": "Confidence",

                    "latitude": "Latitude",

                    "longitude": "Longitude",

                    "timestamp": "Date and Time",

                }

            )



            st.dataframe(

                recent_records,

                use_container_width=True,

                hide_index=True,

                column_config={

                    "Damage Type": st.column_config.TextColumn(

                        "Damage Type",

                        width="small",

                    ),

                    "Confidence": st.column_config.NumberColumn(

                        "Confidence",

                        format="%.2f",

                    ),

                    "Latitude": st.column_config.TextColumn(

                        "Latitude",

                        width="medium",

                    ),

                    "Longitude": st.column_config.TextColumn(

                        "Longitude",

                        width="medium",

                    ),

                    "Date and Time": st.column_config.TextColumn(

                        "Date and Time",

                        width="medium",

                    ),

                },

            )





# =========================================================

# LIVE CAMERA AND PHONE GPS PAGE

# =========================================================



elif page == "📷 Live Camera":

    st.subheader("📷 Live Road Damage Detection")

    st.write(

        "Use your mobile phone camera to detect road damage. "

        "The YOLO model runs on your laptop."

    )

    st.info(

        "Open this app using the current Cloudflare HTTPS link "

        "on your phone. Allow camera and location permissions."

    )



    # Session state setup.

    if "gps_request_id" not in st.session_state:

        st.session_state.gps_request_id = 0



    if "phone_gps" not in st.session_state:

        st.session_state.phone_gps = None



    if "ip_location" not in st.session_state:

        st.session_state.ip_location = None



    # -----------------------------------------------------

    # LOCATION SETTINGS FIRST

    # Use previously received coordinates if available.

    # -----------------------------------------------------



    st.markdown("### 📍 Location Settings")



    location_method = st.radio(

        "Choose location method",

        [

            "Phone GPS",

            "Approximate IP Location",

            "Manual Coordinates",

            "No Location",

        ],

        key="location_method",

    )



    latitude = None

    longitude = None

    gps_accuracy = None

    location_source = "Not available"



    phone_gps = st.session_state.phone_gps



    if location_method == "Phone GPS":

        if phone_gps:

            latitude = phone_gps["latitude"]

            longitude = phone_gps["longitude"]

            gps_accuracy = phone_gps["accuracy"]

            location_source = "Phone GPS"



            st.success(

                f"Phone location ready: "

                f"{latitude:.6f}, {longitude:.6f}"

            )

            st.caption(

                f"GPS accuracy: {gps_accuracy:.2f} metres"

            )

        else:

            st.warning(

                "GPS has not been received yet. "

                "You can start the camera without GPS, "

                "or press Get GPS below."

            )



    elif location_method == "Approximate IP Location":

        if st.button("Get Approximate Location"):

            ip_location = get_ip_location()

            if ip_location:

                st.session_state.ip_location = ip_location

            else:

                st.session_state.ip_location = None

                st.error("Could not get IP location.")



        if st.session_state.ip_location:

            latitude, longitude = st.session_state.ip_location

            location_source = "IP Approximate"

            st.success(

                f"Approximate location: "

                f"{latitude:.6f}, {longitude:.6f}"

            )

            st.caption(

                "IP location is approximate and may not represent "

                "your phone's exact position."

            )



    elif location_method == "Manual Coordinates":

        col1, col2 = st.columns(2)



        with col1:

            latitude = st.number_input(

                "Latitude",

                value=16.700000,

                format="%.6f",

            )



        with col2:

            longitude = st.number_input(

                "Longitude",

                value=74.460000,

                format="%.6f",

            )



        location_source = "Manual"



    else:

        location_source = "Not available"



    if latitude is not None and longitude is not None:

        st.success("Location is ready for camera detection.")

    else:

        st.caption(

            "The camera can run without GPS, but detections "

            "without coordinates cannot be plotted on the map."

        )



    st.divider()



    # -----------------------------------------------------

    # CAMERA IS DISPLAYED BEFORE THE GPS COMPONENT.

    # This is the main fix for the delayed START button.

    # -----------------------------------------------------



    st.markdown("### 🎥 Camera")



    if not MODEL_PATH.exists():

        st.error(

            "Model file missing. Please check that "

            "models/best.pt exists."

        )

    else:

        try:

            model = load_model()



            rtc_config = {

                "iceServers": [

                    {

                        "urls": ["stun:stun.l.google.com:19302"]

                    }

                ]

            }



            webrtc_ctx = webrtc_streamer(

                key="road-damage-camera",

                video_processor_factory=lambda: RoadDamageProcessor(

                    model,

                    latitude,

                    longitude,

                    location_source,

                    gps_accuracy,

                    confidence,

                ),

                rtc_configuration=rtc_config,

                media_stream_constraints={

                    "video": True,

                    "audio": False,

                },

                async_processing=True,

            )



            if webrtc_ctx.state.playing:

                st.success(

                    "Camera is running. AI detection is active."

                )

            else:

                st.warning(

                    "Camera is stopped. Press START to begin."

                )



        except Exception as error:

            st.error(f"Camera error: {error}")



    st.caption(

        "Each damage category is saved at most once every "

        "5 seconds. After receiving new GPS coordinates, stop "

        "and restart the camera so new detections use that location."

    )



    # -----------------------------------------------------

    # GPS REQUEST AFTER CAMERA

    # -----------------------------------------------------



    st.divider()

    st.markdown("### 📍 Phone GPS")



    if st.button("📍 Get GPS", key="request_phone_gps"):

        st.session_state.gps_request_id += 1



    if st.session_state.gps_request_id > 0:

        location = get_geolocation(

            component_key=(

                f"phone_gps_{st.session_state.gps_request_id}"

            )

        )



        if location is None:

            st.info(

                "Waiting for phone GPS. Keep this page open "

                "and allow location permission in your browser."

            )



        elif isinstance(location, dict) and "error" in location:

            error_data = location["error"]



            if isinstance(error_data, dict):

                error_message = error_data.get(

                    "message",

                    str(error_data),

                )

            else:

                error_message = str(error_data)



            st.error(f"GPS Error: {error_message}")



        elif isinstance(location, dict):

            coords = location.get("coords")



            if coords:

                try:

                    latitude_value = float(coords["latitude"])

                    longitude_value = float(coords["longitude"])

                    accuracy_value = float(

                        coords.get("accuracy", 0)

                    )



                    if (

                        -90 <= latitude_value <= 90

                        and -180 <= longitude_value <= 180

                    ):

                        new_gps = {

                            "latitude": latitude_value,

                            "longitude": longitude_value,

                            "accuracy": accuracy_value,

                        }



                        # Save and rerun only when new coordinates arrive.

                        if new_gps != st.session_state.phone_gps:

                            st.session_state.phone_gps = new_gps

                            st.success("Phone GPS location received!")

                            st.write(

                                "Latitude:",

                                f"{latitude_value:.6f}",

                            )

                            st.write(

                                "Longitude:",

                                f"{longitude_value:.6f}",

                            )

                            st.write(

                                "Accuracy:",

                                f"{accuracy_value:.2f} metres",

                            )

                            st.info(

                                "Stop and restart the camera to apply "

                                "these coordinates to new detections."

                            )

                        else:

                            st.success("Phone GPS location is available.")

                            st.write(

                                "Latitude:",

                                f"{latitude_value:.6f}",

                            )

                            st.write(

                                "Longitude:",

                                f"{longitude_value:.6f}",

                            )

                            st.write(

                                "Accuracy:",

                                f"{accuracy_value:.2f} metres",

                            )

                    else:

                        st.error("Invalid GPS coordinates received.")



                except (TypeError, ValueError, KeyError):

                    st.error("Could not read the GPS coordinates.")

            else:

                st.warning(

                    "GPS response did not contain coordinates."

                )





# =========================================================

# DAMAGE MAP PAGE

# =========================================================



elif page == "🗺️ Damage Map":

    st.subheader("📍 All Road Damage Locations")



    if df.empty:

        st.info("No road damage records available.")

    else:

        mapped_df = df.dropna(

            subset=["latitude", "longitude"]

        ).copy()



        mapped_df = mapped_df[

            mapped_df["latitude"].between(-90, 90)

            & mapped_df["longitude"].between(-180, 180)

        ]



        if mapped_df.empty:

            st.warning(

                "No valid GPS coordinates available. "

                "Receive phone GPS and restart the camera before "

                "creating new detections."

            )

        else:

            st.success(

                f"{len(mapped_df)} mapped detection records"

            )



            center_lat = mapped_df["latitude"].mean()

            center_lon = mapped_df["longitude"].mean()



            road_map = folium.Map(

                location=[center_lat, center_lon],

                zoom_start=13,

            )



            for _, row in mapped_df.iterrows():

                lat = float(row["latitude"])

                lon = float(row["longitude"])



                confidence_value = row["confidence"]

                if pd.isna(confidence_value):

                    confidence_text = "Unknown"

                else:

                    confidence_text = f"{confidence_value:.2%}"



                accuracy_value = row.get("gps_accuracy")

                if pd.isna(accuracy_value):

                    accuracy_text = "Not available"

                else:

                    accuracy_text = f"{accuracy_value:.2f} metres"



                popup_html = f"""

                <b>Damage Type:</b> {row['damage_type']}<br>

                <b>Confidence:</b> {confidence_text}<br>

                <b>Latitude:</b> {lat:.6f}<br>

                <b>Longitude:</b> {lon:.6f}<br>

                <b>GPS Accuracy:</b> {accuracy_text}<br>

                <b>Timestamp:</b> {row['timestamp']}

                """



                folium.Marker(

                    location=[lat, lon],

                    popup=folium.Popup(

                        popup_html,

                        max_width=300,

                    ),

                    tooltip=(

                        f"{row['damage_type']} - {confidence_text}"

                    ),

                    icon=folium.Icon(

                        color="red",

                        icon="warning-sign",

                    ),

                ).add_to(road_map)



            st_folium(

                road_map,

                width=None,

                height=500,

                key="all_road_damage_map",

            )



            st.markdown("### Mapped Detection Records")

            st.dataframe(

                mapped_df[

                    [

                        "damage_type",

                        "confidence",

                        "latitude",

                        "longitude",

                        "gps_accuracy",

                        "timestamp",

                    ]

                ],

                use_container_width=True,

                hide_index=True,

            )





# =========================================================

# ANALYTICS PAGE

# =========================================================



elif page == "📊 Analytics":

    st.subheader("📊 Road Damage Analytics")



    if df.empty:

        st.info("No data available for analytics.")

    else:

        category_counts = (

            df["damage_type"]

            .value_counts()

            .reset_index()

        )

        category_counts.columns = ["Damage Type", "Count"]



        st.markdown("### Damage Category Statistics")

        st.bar_chart(

            category_counts.set_index("Damage Type")

        )

        st.dataframe(

            category_counts,

            use_container_width=True,

            hide_index=True,

        )



        st.markdown("### Confidence Distribution")

        st.bar_chart(df["confidence"].dropna())



        st.markdown("### Filter Records")

        all_categories = sorted(

            df["damage_type"].dropna().unique().tolist()

        )



        selected = st.multiselect(

            "Select damage categories",

            options=all_categories,

            default=all_categories,

        )



        filtered = df[df["damage_type"].isin(selected)]



        st.dataframe(

            filtered,

            use_container_width=True,

            hide_index=True,

        )



        st.download_button(

            "⬇️ Download CSV Report",

            data=filtered.to_csv(index=False).encode("utf-8"),

            file_name="road_damage_report.csv",

            mime="text/csv",

        )





# =========================================================

# DETECTION GALLERY - ALL SAVED DAMAGE IMAGES

# =========================================================



elif page == "🖼️ Detection Gallery":

    st.subheader("🖼️ Road Damage Image Gallery")



    st.divider()

    if df.empty:

        st.info("No saved detection images.")

    else:

        gallery = df[

            df["image_path"].fillna("").apply(

                lambda path: bool(path) and Path(path).is_file()

            )

        ].copy()



        if gallery.empty:

            st.warning("No image files found in the detections folder.")

        else:

            st.success(f"Total saved images: {len(gallery)}")



            # Display every saved image, grouped by damage category.

            damage_categories = CLASS_NAMES + [

                category

                for category in gallery["damage_type"].dropna().unique()

                if category not in CLASS_NAMES

            ]



            for damage_type in damage_categories:

                category_images = gallery[

                    gallery["damage_type"] == damage_type

                ]



                if category_images.empty:

                    continue



                st.divider()

                st.subheader(f"Damage Type: {damage_type}")

                st.write(

                    f"Images found: {len(category_images)}"

                )



                cols = st.columns(3)



                for index, (_, row) in enumerate(

                    category_images.iterrows()

                ):

                    with cols[index % 3]:

                        st.image(

                            str(row["image_path"]),

                            use_container_width=True,

                        )



                        confidence_value = row["confidence"]

                        if pd.notna(confidence_value):

                            st.caption(

                                f"Confidence: {confidence_value:.2%}"

                            )



                        st.caption(str(row["timestamp"]))





## =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #e5e7eb;">
        <small>
            AI-Based Road Damage Detection and Geographical Mapping System
            <br>
            Powered by YOLO11, Python, OpenCV and Streamlit
        </small>
    </div>
    """,
    unsafe_allow_html=True
)