#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Synchronous inference script with background Cosmos VLM failure monitoring.

Extends eval_sync.py by running a CosmosMonitor daemon thread that:
  - Samples gripper-camera frames at 4 FPS from the robot observation
  - Every 10 seconds encodes the buffer as an H.264 video and sends it to a
    vLLM server (Cosmos-Reason2) for failure detection
  - If the model returns a string containing the configured failure phrase
    (default: "Failure Detected"), the main task is automatically switched to
    Task 3 ("Move the robot to the home position")

The monitor runs entirely on a background daemon thread.  The only work added
to the hot policy loop is:
  1. A numpy array copy + lock acquire/release to push the latest frame
     (~5 µs) — done AFTER send_action so it never delays actions.
  2. A threading.Event.is_set() check (~100 ns) per iteration.

Usage:
    python eval_sync_cosmos.py \\
        --policy.path=staudi25/pi05_lego_maximus_plus_aug \\
        --policy.device=cuda \\
        --robot.type=so101_follower \\
        --robot.port=/dev/follower \\
        --robot.id=videron_follower \\
        --robot.cameras="{desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \\
        --robot.calibration_dir=/Experiments \\
        --task="Put all the legos on the table in the blue bowl" \\
        --duration=1000 \\
        --cosmos_enabled=true \\
        --cosmos_camera_key=gripper \\
        --cosmos_server_url=http://my_vllm_server:8000/v1/chat/completions
"""

import base64
import io
import logging
import select
import sys
import threading
import time
from dataclasses import dataclass, field

import av
import cv2
import numpy as np
import requests
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- TASK CONFIGURATION ---
TASK_MAP = {
    "1": "Put the red lego in the blue bowl",
    "2": "Put all the legos on the table in the blue bowl",
    "3": "Move the robot to the home position",
}
current_task = TASK_MAP["1"]


def check_for_input(current_val: str) -> str:
    """Non-blocking stdin check for task switching."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if line in TASK_MAP:
            new_task = TASK_MAP[line]
            print(f">>> SWITCHING TO: {new_task}")
            return new_task
        else:
            print(f"Unknown command '{line}'. Available: {list(TASK_MAP.keys())}")
    return current_val


# ---------------------------------------------------------------------------
# Cosmos VLM failure monitor
# ---------------------------------------------------------------------------

class CosmosMonitor:
    """
    Background daemon thread that samples gripper frames, batches them into a
    10-second H.264 video, and queries a vLLM server for failure detection.

    Thread safety
    -------------
    push_frame()     — called from the main thread; acquires _frame_lock briefly.
    check_failure()  — called from the main thread; non-blocking Event.is_set().
    _run()           — runs on a daemon thread; never touches the robot or policy.
    """

    def __init__(
        self,
        camera_key: str,
        server_url: str,
        model_name: str,
        sample_fps: int = 4,
        window_seconds: int = 15,
        failure_phrase: str = "no",
        frame_w: int = 640,
        frame_h: int = 480,
    ):
        self.camera_key = camera_key
        self.server_url = server_url
        self.model_name = model_name
        self.sample_fps = sample_fps
        self.window_seconds = window_seconds
        self.frame_count = sample_fps * window_seconds
        self.failure_phrase = failure_phrase.lower()
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Shared state — only _latest_frame is protected by a lock.
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        # Signals failure to the main thread; cleared after consumption.
        self._failure_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API (called from main thread)
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Store the most-recent gripper frame.  Called after send_action() so it
        never adds latency to the action selection path.  The copy is ~5 µs for
        a 640×480 uint8 array.
        """
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def check_failure(self) -> bool:
        """
        Non-blocking check (~100 ns).  Returns True and clears the event the
        first time it is called after a failure is detected.
        """
        if self._failure_event.is_set():
            self._failure_event.clear()
            return True
        return False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="CosmosMonitor", daemon=True
        )
        self._thread.start()
        logger.info(
            f"[Cosmos] Monitor started — camera='{self.camera_key}', "
            f"{self.sample_fps} fps, {self.window_seconds}s window, "
            f"failure_phrase='{self.failure_phrase}'"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        frame_buffer: list[np.ndarray] = []
        sample_interval = 1.0 / self.sample_fps
        last_sample_time = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_sample_time >= sample_interval:
                last_sample_time = now  # always advance to prevent busy-wait when no frame yet
                with self._frame_lock:
                    frame = self._latest_frame
                if frame is not None:
                    resized = cv2.resize(
                        frame, (self.frame_w, self.frame_h),
                        interpolation=cv2.INTER_AREA
                    )
                    frame_buffer.append(resized)

                if len(frame_buffer) >= self.frame_count:
                    logger.info(
                        f"[Cosmos] Collected {self.frame_count} frames "
                        f"({self.window_seconds}s). Running inference..."
                    )
                    t0 = time.monotonic()
                    result = self._infer(frame_buffer)
                    elapsed = time.monotonic() - t0
                    logger.info(f"[Cosmos] Inference took {elapsed:.2f}s  →  {result!r}")

                    if result and self.failure_phrase in result.lower():
                        logger.warning(
                            "[Cosmos] FAILURE DETECTED — requesting switch to Task 3"
                        )
                        self._failure_event.set()

                    frame_buffer = []
            else:
                # Sleep until next sample is due (avoids busy-waiting).
                time.sleep(min(0.02, sample_interval - (now - last_sample_time)))

    def _infer(self, frames: list[np.ndarray]) -> str | None:
        try:
            b64 = self._encode_video(frames)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Has the gripper picked up and placed an object in the blue bowl? Think carefully. Answer yes or no."
                                ),
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                            },
                        ],
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            response = requests.post(self.server_url, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning(f"[Cosmos] Inference error (non-fatal): {exc}")
            return None

    def _encode_video(self, frames: list[np.ndarray]) -> str:
        """Encode a list of RGB uint8 frames to a base64 H.264 MP4."""
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=self.sample_fps)
            stream.width = frames[0].shape[1]
            stream.height = frames[0].shape[0]
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_gripper_frame(obs: dict, camera_key: str) -> np.ndarray | None:
    """
    Extract a gripper camera frame from the raw robot observation dict.

    lerobot robots return observation dicts with keys that follow the pattern
    used in robot.observation_features.  Camera images are typically stored
    under "images.<camera_name>" (e.g. "images.gripper").  We try several
    common patterns so this works regardless of lerobot version.
    """
    candidates = [
        f"images.{camera_key}",   # standard lerobot key
        camera_key,                # bare key fallback
    ]
    for key in candidates:
        frame = obs.get(key)
        if frame is not None and isinstance(frame, np.ndarray):
            return frame
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig(HubMixin):
    """Configuration for synchronous inference with real robots."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    duration: float = 30.0
    fps: float = 30.0
    task: str = field(default="", metadata={"help": "Default task to execute"})

    # Rerun visualization
    visualize: bool = False
    rerun_session_name: str = "eval_sync"
    rerun_ip: str | None = None
    rerun_port: int | None = None
    compress_images: bool = False

    # Cosmos VLM failure monitor
    cosmos_enabled: bool = False
    cosmos_camera_key: str = "gripper"
    cosmos_server_url: str = "http://my_vllm_server:8000/v1/chat/completions"
    cosmos_model_name: str = "/Experiments/cosmos-reason2-2b_v1208-fp8-static-kv8"
    cosmos_failure_phrase: str = "no"
    cosmos_sample_fps: int = 4
    cosmos_window_seconds: int = 15
    cosmos_frame_w: int = 640
    cosmos_frame_h: int = 480

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path=...)")

        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def main(cfg: SyncConfig):
    init_logging()
    logger.info("Starting synchronous inference")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # --- Load policy ---
    logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(cfg.policy.pretrained_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, cfg.policy.pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy = policy.to(cfg.policy.device)
    policy.eval()
    policy.reset()
    logger.info(f"Policy loaded on {cfg.policy.device}")

    # --- Load pre/post processors ---
    logger.info(f"Loading processors from {cfg.policy.pretrained_path}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )
    logger.info("Processors loaded")

    # --- Connect robot ---
    logger.info(f"Connecting robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    if cfg.visualize:
        init_rerun(
            session_name=cfg.rerun_session_name,
            ip=cfg.rerun_ip,
            port=cfg.rerun_port,
        )
        logger.info("Rerun visualization initialized")

    # --- Start Cosmos monitor (optional) ---
    cosmos: CosmosMonitor | None = None
    if cfg.cosmos_enabled:
        cosmos = CosmosMonitor(
            camera_key=cfg.cosmos_camera_key,
            server_url=cfg.cosmos_server_url,
            model_name=cfg.cosmos_model_name,
            sample_fps=cfg.cosmos_sample_fps,
            window_seconds=cfg.cosmos_window_seconds,
            failure_phrase=cfg.cosmos_failure_phrase,
            frame_w=cfg.cosmos_frame_w,
            frame_h=cfg.cosmos_frame_h,
        )
        cosmos.start()

    global current_task
    if cfg.task:
        current_task = cfg.task

    # Task revert state for cosmos-triggered switches
    cosmos_revert_task: str | None = None   # task to return to after Task 3
    cosmos_revert_at: float | None = None   # monotonic time to revert

    action_interval = 1.0 / cfg.fps
    start_time = time.time()
    step = 0

    logger.info(f"Running for {cfg.duration}s at {cfg.fps} Hz — task: '{current_task}'")
    logger.info(f"Type a task key {list(TASK_MAP.keys())} + Enter to switch tasks")
    if cosmos is not None:
        logger.info(
            f"Cosmos monitor active on camera '{cfg.cosmos_camera_key}' — "
            f"will switch to Task 3 on '{cfg.cosmos_failure_phrase}'"
        )

    try:
        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            t0 = time.perf_counter()

            # Non-blocking task switch from keyboard
            current_task = check_for_input(current_task)

            # Non-blocking task switch from Cosmos (~100 ns)
            if cosmos is not None and cosmos.check_failure():
                cosmos_revert_task = current_task
                cosmos_revert_at = time.monotonic() + 5.0
                current_task = TASK_MAP["3"]
                logger.warning(
                    f"[Cosmos] Failure detected — switching to Task 3, "
                    f"will revert to '{cosmos_revert_task}' in 5s"
                )

            # Revert from Task 3 back to prior task after 3 seconds
            if cosmos_revert_at is not None and time.monotonic() >= cosmos_revert_at:
                current_task = cosmos_revert_task
                logger.info(f"[Cosmos] Reverting task to: '{current_task}'")
                cosmos_revert_task = None
                cosmos_revert_at = None

            # --- Observation ---
            obs = robot.get_observation()

            if cfg.visualize:
                import rerun as rr
                rr.set_time_sequence("step", step)
                log_rerun_data(observation=obs, compress_images=cfg.compress_images)

            obs_processed = robot_observation_processor(obs)
            obs_dict = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

            for name in obs_dict:
                obs_dict[name] = torch.from_numpy(obs_dict[name])
                if "image" in name:
                    obs_dict[name] = obs_dict[name].type(torch.float32) / 255
                    obs_dict[name] = obs_dict[name].permute(2, 0, 1).contiguous()
                obs_dict[name] = obs_dict[name].unsqueeze(0).to(cfg.policy.device)

            obs_dict["task"] = [current_task]
            obs_dict["robot_type"] = robot.name if hasattr(robot, "name") else ""

            # --- Preprocess ---
            batch = preprocessor(obs_dict)

            # --- Synchronous inference ---
            action = policy.select_action(batch)  # (1, action_dim)

            # --- Postprocess & send ---
            action_postprocessed = postprocessor(action).squeeze(0).cpu()  # (action_dim,)
            action_dict = {
                key: action_postprocessed[i].item()
                for i, key in enumerate(robot.action_features)
            }
            action_processed = robot_action_processor((action_dict, None))
            robot.send_action(action_processed)

            # --- Push gripper frame to Cosmos monitor AFTER send_action ---
            # This is intentionally placed last so it never delays action delivery.
            if cosmos is not None:
                gripper_frame = _extract_gripper_frame(obs, cfg.cosmos_camera_key)
                if gripper_frame is not None:
                    cosmos.push_frame(gripper_frame)

            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"[MAIN] step={step}  elapsed={elapsed:.1f}s  task='{current_task}'")

            # Pace the loop to target fps
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, action_interval - dt - 0.001))

    finally:
        logger.info("Shutting down")
        shutdown_event.set()
        if cosmos is not None:
            cosmos.stop()
        robot.disconnect()
        logger.info(f"Robot disconnected — total steps: {step}")


if __name__ == "__main__":
    main()
