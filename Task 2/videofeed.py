import cv2
import requests
import numpy as np

# URL of the video stream from the Flask server
stream_url = "http://192.168.174.93:5000/video"

# Open the video stream
cap = requests.get(stream_url, stream=True)

# Read the stream frame by frame
byte_stream = b""
for chunk in cap.iter_content(chunk_size=1024):
    byte_stream += chunk
    a = byte_stream.find(b'\xff\xd8')  # Start of JPEG frame
    b = byte_stream.find(b'\xff\xd9')  # End of JPEG frame
    
    if a != -1 and b != -1:
        jpg = byte_stream[a:b+2]  # Extract the JPEG frame
        byte_stream = byte_stream[b+2:]  # Remove processed bytes
        
        # Convert to NumPy array and decode
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Display the frame
        cv2.imshow("Live Video Stream", frame)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

cv2.destroyAllWindows()
