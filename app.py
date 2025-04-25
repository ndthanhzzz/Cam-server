from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import toml  # Import thư viện toml
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Cấu hình secret key cho session
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

global active_filter
active_filter = None
global right_finger_count_val
global left_finger_count_val
right_finger_count_val = 0
left_finger_count_val = 0

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def load_users(config_file='config.txt'):
    """Đọc thông tin người dùng từ file cấu hình TOML."""
    try:
        with open(config_file, 'r') as f:
            config = toml.load(f)
        if 'users' in config and isinstance(config['users'], dict):
            return config['users']
        return {}
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file cấu hình '{config_file}'")
        return {}
    except toml.TomlDecodeError as e:
        print(f"Lỗi giải mã file TOML '{config_file}': {e}")
        return {}

users = load_users()
print(f"Thông tin người dùng đã tải: {users}") # In ra để kiểm tra

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

def calculate_angle(p1, p2, p3):
    """Tính góc giữa ba điểm."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p1.y - p2.y]) # Lỗi ở đây, đã sửa
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def count_fingers(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                  mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                  mp_hands.HandLandmark.RING_FINGER_TIP,
                  mp_hands.HandLandmark.PINKY_TIP]
    finger_bases = [mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                    mp_hands.HandLandmark.RING_FINGER_MCP,
                    mp_hands.HandLandmark.PINKY_MCP]
    thumb_tip = mp_hands.HandLandmark.THUMB_TIP
    thumb_ip = mp_hands.HandLandmark.THUMB_IP
    thumb_mcp = mp_hands.HandLandmark.THUMB_MCP

    fingers_up = 0

    # Ngón cái: Kiểm tra góc
    angle = calculate_angle(hand_landmarks.landmark[thumb_tip],
                             hand_landmarks.landmark[thumb_ip],
                             hand_landmarks.landmark[thumb_mcp])
    if angle > 160:  # Ngưỡng góc có thể cần điều chỉnh
        fingers_up += 1

    # Các ngón còn lại: Kiểm tra vị trí đốt ngón tay so với đốt gần nhất
    for tip_index, base_index in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[base_index].y:
            fingers_up += 1

    return fingers_up

def get_finger_data():
    global right_finger_count_val
    global left_finger_count_val
    success, frame = camera.read()
    if not success:
        return jsonify({'right': 0, 'left': 0})
    else:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        right_count = 0
        left_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                finger_count = count_fingers(hand_landmarks)

                if hand_label == 'Right':
                    right_count = finger_count
                elif hand_label == 'Left':
                    left_count = finger_count

        right_finger_count_val = right_count
        left_finger_count_val = left_count
        return jsonify({'right': right_count, 'left': left_count})

def generate_finger_count_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()
    hands.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Tên đăng nhập hoặc mật khẩu không đúng.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))

def login_required(func):
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/index')
@login_required
def filtered_feed():
    return Response(generate_filtered_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_detection')
@login_required
def face_detection_page():
    return render_template('face_detection.html')

@app.route('/face_detection_feed')
@login_required
def face_detection_feed():
    return Response(generate_face_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finger_count')
@login_required
def finger_count_page():
    return render_template('finger_count.html')

@app.route('/finger_count_feed')
@login_required
def finger_count_feed():
    return Response(generate_finger_count_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finger_data')
@login_required
def finger_data():
    return get_finger_data()

@app.route('/apply_filter', methods=['POST'])
@login_required
def apply_filter_route():
    global active_filter
    filter_type = request.form.get('filter')
    active_filter = filter_type
    return "Filter applied"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)