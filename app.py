from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

global active_filter
active_filter = None

def apply_filter(frame, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    # Thêm các bộ lọc khác ở đây
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if active_filter:
                frame = apply_filter(frame, active_filter)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Đảm bảo giải phóng camera khi generator dừng
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/apply_filter', methods=['POST'])
def apply_filter_route():
    global active_filter
    filter_type = request.form.get('filter')
    active_filter = filter_type
    return "Filter applied"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)