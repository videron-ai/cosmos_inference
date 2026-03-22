import cv2
import requests
import base64
import time

# Configuration
SERVER_URL = "http://my_vllm_server:8000/v1/chat/completions"
MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

cap = cv2.VideoCapture(0) # 0 is usually the default webcam

print("Starting webcam stream... Press 'q' to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        # Define the target dimensions for 720p (width=1280, height=720)
        # Note: cv2.resize takes (width, height) as the second argument, not (height, width)
        new_dimensions = (1280,720)

        # Resize the frame using cv2.resize
        # INTER_AREA is generally recommended for downsampling, while INTER_CUBIC or INTER_LINEAR are good for upsampling
        rame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)

        if not ret: break

        # Show the live feed
        #cv2.imshow('Cosmos Reason 2 Live Feed', frame)

        # Basic inference loop
        # In a production app, run this in a separate thread
        current_time = time.time()
        
        # Prepare the payload (Sending a single frame as an image_url)
        # For temporal reasoning, you can send a list of recent frames
        base64_image = encode_frame(frame)
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the scene in one sentence"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 512
        }

        try:
            response = requests.post(SERVER_URL, json=payload)
            result = response.json()
            print(f"\n[Cosmos]: {result['choices'][0]['message']['content']}")
        except Exception as e:
            print(f"Error: {e}")

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

finally:
    cap.release()
    #cv2.destroyAllWindows()
