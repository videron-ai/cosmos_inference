# cosmos_inference

Synchronous robot policy inference with background [Cosmos-Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-2B) VLM failure monitoring, built on [LeRobot](https://github.com/huggingface/lerobot).

## Overview

`eval_sync_cosmos.py` runs a LeRobot policy on a real robot at a fixed control frequency while a background daemon thread watches a gripper camera for task failures. When the VLM detects a failure it automatically switches the active task (e.g. to a home-position recovery motion) and then reverts after a configurable delay — all without interrupting the hot action loop.

### How it works

```
Main thread (30 Hz)                    CosmosMonitor daemon thread
──────────────────────────────────     ──────────────────────────────────────
get_observation()                      sample latest frame at 4 FPS
preprocess → policy.select_action()    accumulate into 15-second ring buffer
send_action()                     ───> push_frame() (~5 µs, lock + copy)
check_failure() (~100 ns)              encode buffer → H.264 MP4
                                       POST video to vLLM (Cosmos-Reason2)
                                       if "no" in response → set failure event
```

The only cost added to the policy loop is a `numpy` array copy after `send_action` and a non-blocking `Event.is_set()` check — neither delays action delivery.

---

## Files

| File | Description |
|---|---|
| `eval_sync_cosmos.py` | Main inference script with integrated Cosmos monitor |
| `cosmos_video_client.py` | Standalone script: webcam → sliding video window → Cosmos |
| `cosmos_streaming_client.py` | Standalone script: webcam → per-frame JPEG → Cosmos |

---

## Requirements

- Python 3.10+
- [LeRobot](https://github.com/huggingface/lerobot) (with robot drivers for your hardware)
- A vLLM server serving a Cosmos-Reason2 model with video support
- `av`, `opencv-python`, `requests`, `torch`

```bash
pip install av opencv-python requests torch
```

---

## Usage

```bash
python eval_sync_cosmos.py \
    --policy.path=staudi25/pi05_lego_maximus_plus_aug \
    --policy.device=cuda \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=videron_follower \
    --robot.cameras="{
        desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'},
        gripper:  {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}
    }" \
    --robot.calibration_dir=/Experiments \
    --task="Put all the legos on the table in the blue bowl" \
    --duration=1000 \
    --cosmos_enabled=true \
    --cosmos_camera_key=gripper \
    --cosmos_server_url=http://my_vllm_server:8000/v1/chat/completions
```

---

## Configuration

### Core

| Flag | Default | Description |
|---|---|---|
| `--policy.path` | *(required)* | HuggingFace repo or local path to the LeRobot policy |
| `--policy.device` | — | PyTorch device (`cuda`, `cpu`) |
| `--robot.type` | *(required)* | Robot driver type (e.g. `so101_follower`) |
| `--task` | `""` | Initial task string passed to the policy |
| `--duration` | `30.0` | Total run time in seconds |
| `--fps` | `30.0` | Control frequency in Hz |

### Cosmos VLM monitor

| Flag | Default | Description |
|---|---|---|
| `--cosmos_enabled` | `false` | Enable the background failure monitor |
| `--cosmos_camera_key` | `gripper` | Camera key to sample from the observation dict |
| `--cosmos_server_url` | `http://my_vllm_server:8000/v1/chat/completions` | vLLM endpoint |
| `--cosmos_model_name` | `/Experiments/cosmos-reason2-2b_v1208-fp8-static-kv8` | Model name/path sent in the request |
| `--cosmos_failure_phrase` | `no` | String to watch for in VLM response (case-insensitive) |
| `--cosmos_sample_fps` | `4` | Frame sampling rate for the video buffer |
| `--cosmos_window_seconds` | `15` | Length of each video clip sent to the VLM |
| `--cosmos_frame_w` | `640` | Frame width before encoding |
| `--cosmos_frame_h` | `480` | Frame height before encoding |

### Rerun visualization (optional)

| Flag | Default | Description |
|---|---|---|
| `--visualize` | `false` | Enable Rerun telemetry |
| `--rerun_session_name` | `eval_sync` | Rerun session name |
| `--rerun_ip` | `None` | Rerun server IP |
| `--rerun_port` | `None` | Rerun server port |
| `--compress_images` | `false` | JPEG-compress images before logging |

---

## Task switching

### Keyboard (runtime)

Type a task key and press Enter while the script is running:

| Key | Task |
|---|---|
| `1` | Put the red lego in the blue bowl |
| `2` | Put all the legos on the table in the blue bowl |
| `3` | Move the robot to the home position |

### Automatic (Cosmos-triggered)

When the VLM response contains the configured `failure_phrase`:

1. The current task is saved.
2. Task 3 ("Move the robot to the home position") becomes active.
3. After **5 seconds**, the original task is restored.

The revert delay can be adjusted in `main()` (`cosmos_revert_at = time.monotonic() + 5.0`).

---

## Cosmos prompt

The monitor sends the following prompt to the VLM:

> *"Has the gripper picked up and placed an object in the blue bowl? Think carefully. Answer yes or no."*

A response containing `"no"` (the default `failure_phrase`) triggers the recovery switch. Change `--cosmos_failure_phrase` and the prompt text in `CosmosMonitor._infer()` to adapt this to a different task or failure definition.

---

## vLLM server setup

Cosmos-Reason2 must be served with video support enabled. Example launch:

```bash
vllm serve nvidia/Cosmos-Reason2-2B \
    --dtype auto \
    --max-model-len 8192 \
    --limit-mm-per-prompt video=1
```

The script sends video as a base64-encoded H.264 MP4 via the `video_url` content type in the OpenAI-compatible chat API.
