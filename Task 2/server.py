import cv2
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    # Initialize the camera here so that it’s created in the correct process
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Couldn't read frame")
            continue  # Skip this iteration if frame reading fails

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Frame encoding failed")
            continue

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.174.93',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False)

