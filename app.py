import os
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app = Flask(__name__)

# ---------------- PATHS ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascades", "haarcascade_eye.xml")

# ---------------- CAMERA ---------------- #
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not accessible")
    exit()

# ---------------- EYE CASCADE ---------------- #
eye_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if eye_cascade.empty():
    raise RuntimeError("❌ Error loading haarcascade file")
# ---------------- SAFE ENCODING ---------------- #
def load_encoding(filename, name):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ File not found: {filename}")
        return None
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"❌ No face found in {name}")
        return None
    return encodings[0]

# Load images safely
tanzim_face_encoding = load_encoding("BeingTanzim.jpg", "Tanzim")
yusra_face_encoding = load_encoding("Yusra.jpg", "Yusra")
aziz_face_encoding = load_encoding("Aziz.jpg", "Aziz")
armaan_face_encoding = load_encoding("Armaan.jpg", "Armaan")
makun_face_encoding = load_encoding("Makun.jpg", "Makun")

known_face_encodings = []
known_face_names = []

if tanzim_face_encoding is not None:
    known_face_encodings.append(tanzim_face_encoding)
    known_face_names.append("Tanzim")

if yusra_face_encoding is not None:
    known_face_encodings.append(yusra_face_encoding)
    known_face_names.append("Yusra Cutie")

if aziz_face_encoding is not None:
    known_face_encodings.append(aziz_face_encoding)
    known_face_names.append("Marriage Ready")

if armaan_face_encoding is not None:
    known_face_encodings.append(armaan_face_encoding)
    known_face_names.append("Baddie Paglu")

if makun_face_encoding is not None:
    known_face_encodings.append(makun_face_encoding)
    known_face_names.append("Makun Chachu")

process_this_frame = True

# ---------------- FRAME GENERATOR ---------------- #
def gen_frames():
    global process_this_frame

    while True:
        success, frame = camera.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = []
        face_names = []

        if process_this_frame:
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )

                    for face_encoding in face_encodings:
                        name = "Unknown"

                        if len(known_face_encodings) > 0:
                            matches = face_recognition.compare_faces(
                                known_face_encodings, face_encoding
                            )
                            face_distances = face_recognition.face_distance(
                                known_face_encodings, face_encoding
                            )

                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                        face_names.append(name)

            except Exception as e:
                print("⚠️ Frame error:", e)
                continue

        process_this_frame = not process_this_frame

        # ---------------- DRAW ---------------- #
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Face box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # -------- EYE DETECTION -------- #
            face_roi = frame[top:bottom, left:right]

            if face_roi.size != 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                eyes = eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi, (ex, ey),
                                  (ex + ew, ey + eh), (255, 0, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ---------------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- RUN ---------------- #
if __name__ == '__main__':
    app.run(debug=True)