import cv2
import time
import os
import threading
import base64
import json

import numpy as np


class ZmqVideoCapture:
    """cv2.VideoCapture look-alike for a camera shared over ZMQ (lerobot_franka/cameras/realsense_server.py),
    so Fabrica and LeRobot can use the same camera at the same time.

    Frames arrive as RGB JPEGs in LeRobot's ZMQ camera format. read() blocks for the next frame like
    cv2.VideoCapture and returns it as BGR, resized to the width/height passed to set().
    """

    def __init__(self, url, timeout=1.0):
        import zmq
        self._zmq = zmq
        self._timeout = timeout
        self._size = {}
        self._lock = threading.Lock()
        self._frame = None
        self._new_frame = threading.Event()
        self._stop = threading.Event()

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.connect(url)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        poller = self._zmq.Poller()
        poller.register(self._sock, self._zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(100):
                continue
            images = json.loads(self._sock.recv_string()).get("images", {})
            if not images:
                continue
            jpg = np.frombuffer(base64.b64decode(next(iter(images.values()))), np.uint8)
            frame = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
            with self._lock:
                self._frame = frame
            self._new_frame.set()

    def isOpened(self):
        return not self._stop.is_set()

    def set(self, prop, value):
        self._size[prop] = int(value)
        return True

    def read(self):
        if not self._new_frame.wait(self._timeout):
            return False, None
        with self._lock:
            frame = self._frame
            self._new_frame.clear()
        width = self._size.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._size.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width and height and frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return True, frame

    def release(self):
        self._stop.set()
        self._thread.join()
        self._sock.close(linger=0)
        self._ctx.term()


class Camera:
    def __init__(self, camera_index=0, warmup_frames=5, warmup_delay=1.0,
                 init_timeout=10.0, retry_interval=0.5, verbose=False):
        self.camera_index = camera_index
        self.warmup_frames = warmup_frames
        self.warmup_delay = warmup_delay
        self.verbose = verbose

        self.cap = None
        self.video_writer = None
        self.video_path = None
        self.fps = 30

        self._recording_thread = None
        self._stop_recording = threading.Event()

        self._initialize_camera(init_timeout, retry_interval)

    def _log(self, level, message):
        if self.verbose:
            print(f"[{level}] {message}")

    def _initialize_camera(self, timeout, retry_interval):
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.cap:
                self.cap.release()

            if isinstance(self.camera_index, str) and self.camera_index.startswith("tcp://"):
                self.cap = ZmqVideoCapture(self.camera_index)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            if not self.cap.isOpened():
                self._log("WARN", f"Failed to open camera at index {self.camera_index}. Retrying...")
                time.sleep(retry_interval)
                continue

            # Discard initial frames to flush buffer
            time.sleep(self.warmup_delay)
            for _ in range(self.warmup_frames):
                self.cap.read()

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._log("INFO", "Camera initialized successfully.")
                return

            self._log("WARN", "Camera opened but failed to return valid frame. Retrying...")
            time.sleep(retry_interval)

        raise RuntimeError(f"Failed to initialize camera at index {self.camera_index} within {timeout} seconds.")

    def _flush_camera_buffer(self):
        """ Read and discard a few frames to clear buffer. """
        for _ in range(self.warmup_frames):
            self.cap.read()
        self._log("INFO", "Camera buffer flushed.")

    def _record_loop(self):
        self._log("INFO", "Recording thread started.")
        while not self._stop_recording.is_set():
            ret, frame = self.cap.read()
            if ret and self.video_writer:
                self.video_writer.write(frame)
            time.sleep(1.0 / self.fps)
        self._log("INFO", "Recording thread stopped.")

    def start_video_recording(self, save_path, fps=30):
        if self._recording_thread and self._recording_thread.is_alive():
            self._log("WARN", "Recording already in progress.")
            return

        # Ensure the writer is fully released before starting a new recording
        self.stop_video_recording()

        # Flush the buffer to avoid old frames
        self._flush_camera_buffer()

        # Initialize video writer
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read initial frame for video recording.")

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        self.video_path = save_path
        self.fps = fps
        self._stop_recording.clear()
        self._recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._recording_thread.start()

        self._log("INFO", f"Started video recording to {save_path}")

    def stop_video_recording(self):
        if self._recording_thread and self._recording_thread.is_alive():
            self._stop_recording.set()
            self._recording_thread.join()
            self._recording_thread = None

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self._log("INFO", f"Stopped recording. Video saved to {self.video_path}")

    def release(self):
        self.stop_video_recording()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self._log("INFO", "Camera released.")


if __name__ == "__main__":
    cam = Camera(camera_index=0, verbose=True)

    try:
        cam.start_video_recording("async_output.mp4", fps=20)
        print("Recording... doing other stuff meanwhile")
        time.sleep(5)  # Simulate some task
        cam.stop_video_recording()
    finally:
        cam.release()
