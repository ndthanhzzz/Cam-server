from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Ví dụ đường dẫn trên Linux


global active_filter
active_filter = None

def apply_filter(frame, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

def generate_filtered_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if active_filter:
                frame = apply_filter(frame, active_filter)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

def generate_face_detection_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def filtered_feed():
    return Response(generate_filtered_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_detection')
def face_detection_page():
    return render_template('face_detection.html')

@app.route('/face_detection_feed')
def face_detection_feed():
    return Response(generate_face_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/apply_filter', methods=['POST'])
def apply_filter_route():
    global active_filter
    filter_type = request.form.get('filter')
    active_filter = filter_type
    return "Filter applied"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)