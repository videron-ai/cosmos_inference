import io
import cv2
import av
import requests
import base64
import time

# Configuration
SERVER_URL = "http://my_vllm_server:8000/v1/chat/completions"
#MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
MODEL_NAME = "/Experiments/cosmos-reason2-2b_v1208-fp8-static-kv8"

SAMPLE_FPS = 4
WINDOW_SECONDS = 10
FRAME_COUNT = SAMPLE_FPS * WINDOW_SECONDS  # 40 frames


def frames_to_b64_video(frames, fps):
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def infer(frames, width, height):
    try:
        b64 = frames_to_b64_video(frames, SAMPLE_FPS)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is there a person present answer yes or no"},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                    ],
                },
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        response = requests.post(SERVER_URL, json=payload)
        result = response.json()
        print(f"\n[Cosmos]: {result['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"Error during inference: {e}")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Read one frame to get actual dimensions
ret, sample = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read from webcam.")

TARGET_W, TARGET_H = 640,480

frame_buffer = []
interval = 1.0 / SAMPLE_FPS
last_sample = time.time()

print(f"Sampling at {SAMPLE_FPS} fps, sending every {WINDOW_SECONDS}s window to Cosmos.")
print("Press Ctrl+C to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_sample >= interval:
            resized = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            #camera is upside down
            resized = cv2.flip(resized, -1)
            frame_buffer.append(resized)
            last_sample = now

            if len(frame_buffer) >= FRAME_COUNT:
                print(f"\nCollected {FRAME_COUNT} frames ({WINDOW_SECONDS}s). Sending to model...")
                t0 = time.time()
                infer(frame_buffer, TARGET_W, TARGET_H)
                print(f"[Inference took {time.time() - t0:.2f}s]")
                frame_buffer = []

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    cap.release()
