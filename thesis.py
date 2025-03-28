from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_socketio import SocketIO
from picamera2 import Picamera2
from ultralytics import YOLO
from io import BytesIO
from gtts import gTTS
import cv2
import os

app = Flask(__name__)

# Load YOLOv8 model
model_path = r"best.pt"  # Make sure the path is correct
model = YOLO(model_path)

# Object-color mapping (same as before)
object_colors = {
    "APPLE": ((0, 0, 255), "Red"),
    "BANANA": ((0, 255, 255), "Yellow"),
    "BITTER MELON": ((0, 255, 0), "Green"),
    "BROCCOLI": ((0, 255, 0), "Green"),
    "CIRCLE": ((0, 255, 0), "Green"),
    "CORN": ((0, 255, 255), "Yellow"),
    "EGGPLANT": ((128, 0, 128), "Purple"),
    "GRAPES": ((128, 0, 128), "Purple"),
    "MUSHROOM": ((42, 42, 165), "Brown"),
    "ORANGE": ((0, 165, 255), "Orange"),
    "OVAL": ((255, 0, 0), "Blue"),
    "PEAR": ((0, 255, 255), "Yellow"),
    "PUMPKIN": ((0, 165, 255), "Orange"),
    "RECTANGLE": ((0, 255, 255), "Yellow"),
    "SQUARE": ((0, 255, 255), "Purple"),
    "STAR": ((0, 165, 255), "Orange"),
    "STRAWBERRY": ((0, 0, 255), "Red"),
    "TOMATO": ((0, 0, 255), "Red"),
    "TRIANGLE": ((255, 105, 180), "Pink"),
    "WATERMELON": ((0, 255, 0), "Green"),
}

# Object detection
# cap = cv2.VideoCapture(1)
tuning = Picamera2.load_tuning_file("imx477_noir.json")
camera = Picamera2(tuning=tuning)
camera.configure(camera.create_preview_configuration(main={"format": 'RGB888', "size": (1280, 1280)}))
camera.set_controls({"Brightness": 0.20, "Saturation":1.1})  # Adjust the value as needed

camera.start()

detected_objects_global = []  # Store detected objects globally

def generate_frames():
    global detected_objects_global
    while True:
        frame = camera.capture_array()
        
        results = model.predict(frame, conf=0.6)
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_object = model.names[class_id].upper()
                (bgr_color, color_name) = object_colors.get(detected_object, ((255, 255, 255), "WHITE"))
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                label = f"{detected_object}, {color_name.upper()}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
                detected_objects.append({"object": detected_object, "color": color_name})

        detected_objects_global = detected_objects  # Update global variable

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_objects', methods=['GET'])
def get_detected_objects():
    return jsonify(detected_objects_global)  # Return the global detected objects


@app.route('/speak', methods=['POST'])
def speak():
    print("OK")
    result_string = ", ".join([f"{obj['object']}, ({obj['color']})" for obj in detected_objects_global])
    if result_string:
        # Use gTTS to convert the text to speech
        tts = gTTS(text=result_string, lang='en')
        
        # Save the speech to a BytesIO object to avoid saving as a file
        speech_io = BytesIO()
        tts.write_to_fp(speech_io)
        speech_io.seek(0)
        
        # Return the audio file
        return send_file(speech_io, mimetype='audio/mp3', as_attachment=True, download_name="speech.mp3")
    
    return jsonify({"message": "No text provided"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
