"""Minimal scrcpy client using ADB and PyAV to stream Android screen."""

import argparse
import os
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import av
import numpy as np
import pygame

CODECS = {
    0x68323634: "h264",
    0x68323635: "hevc",
    0x00617631: "av1",
}

HEADER_SIZE = 12
FLAG_CONFIG = 1 << 63
FLAG_KEY_FRAME = 1 << 62
PTS_MASK = FLAG_KEY_FRAME - 1

SERVER_VERSION = "3.3.1"
DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
LOCK_SCREEN_ORIENTATION_UNLOCKED = 0

# Key and action constants (partial)
KEYEVENT_ACTION_DOWN = 0
KEYEVENT_ACTION_UP = 1

ANDROID_KEYCODES = {
    pygame.K_a: 29,
    pygame.K_b: 30,
    pygame.K_c: 31,
    pygame.K_d: 32,
    pygame.K_e: 33,
    pygame.K_f: 34,
    pygame.K_g: 35,
    pygame.K_h: 36,
    pygame.K_i: 37,
    pygame.K_j: 38,
    pygame.K_k: 39,
    pygame.K_l: 40,
    pygame.K_m: 41,
    pygame.K_n: 42,
    pygame.K_o: 43,
    pygame.K_p: 44,
    pygame.K_q: 45,
    pygame.K_r: 46,
    pygame.K_s: 47,
    pygame.K_t: 48,
    pygame.K_u: 49,
    pygame.K_v: 50,
    pygame.K_w: 51,
    pygame.K_x: 52,
    pygame.K_y: 53,
    pygame.K_z: 54,
    pygame.K_0: 7,
    pygame.K_1: 8,
    pygame.K_2: 9,
    pygame.K_3: 10,
    pygame.K_4: 11,
    pygame.K_5: 12,
    pygame.K_6: 13,
    pygame.K_7: 14,
    pygame.K_8: 15,
    pygame.K_9: 16,
    pygame.K_SPACE: 62,
    pygame.K_RETURN: 66,
    pygame.K_BACKSPACE: 67,
}


def read_exact(sock: socket.socket, length: int) -> bytes:
    """Read exactly `length` bytes from the socket."""
    buf = bytearray()
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            raise EOFError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


@dataclass
class ClientConfig:
    """Configuration for the scrcpy client."""

    # pylint: disable=too-many-instance-attributes
    adb: str = "adb"
    server: str = "scrcpy-server-v3.3.1"
    host: str = "127.0.0.1"
    port: int = 27183
    ip: str = "127.0.0.1:5037"
    max_width: int = 1440
    bitrate: int = 8_000_000
    max_fps: int = 0
    flip: bool = False
    stay_awake: bool = True
    lock_screen_orientation: int = LOCK_SCREEN_ORIENTATION_UNLOCKED
    docker: bool = False
    control: bool = True


@dataclass
class ClientState:
    """Dynamic state during runtime."""

    proc: Optional[subprocess.Popen] = None
    last_frame: Optional[np.ndarray] = None
    resolution: Optional[Tuple[int, int]] = None
    device_name: Optional[str] = None
    thread: Optional[threading.Thread] = None
    control_thread: Optional[threading.Thread] = None
    video_sock: Optional[socket.socket] = None
    control_sock: Optional[socket.socket] = None


class Client:
    """Handles the scrcpy server interaction and video decoding."""

    def __init__(self, client_config: ClientConfig) -> None:
        """Initialize the scrcpy client."""
        self.config = client_config
        adb_host, sep, adb_port = client_config.ip.partition(":")
        self.adb_cmd = [client_config.adb]
        if sep:
            self.adb_cmd += ["-H", adb_host, "-P", adb_port]

        self.state = ClientState()
        self.run()

    def _start_server(self) -> None:
        """Start the scrcpy server on the Android device."""
        server_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.config.server)
        subprocess.run(self.adb_cmd + ["push", server_file_path, DEVICE_SERVER_PATH], check=True)
        subprocess.run(self.adb_cmd + ["forward", f"tcp:{self.config.port}", "localabstract:scrcpy"], check=True)

        cmd = self.adb_cmd + [
            "shell",
            f"CLASSPATH={DEVICE_SERVER_PATH}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            SERVER_VERSION,
            "tunnel_forward=true",
            "audio=false",
            f"control={'true' if self.config.control else 'false'}",
            "cleanup=false",
        ]
        if self.config.max_width:
            cmd.append(f"max_size={self.config.max_width}")
        if self.config.bitrate:
            cmd.append(f"video_bit_rate={self.config.bitrate}")
        if self.config.max_fps:
            cmd.append(f"max_fps={self.config.max_fps}")
        if self.config.flip:
            cmd.append("orientation=flip")
        if self.config.stay_awake:
            cmd.append("stay_awake=true")
        if self.config.lock_screen_orientation != LOCK_SCREEN_ORIENTATION_UNLOCKED:
            cmd.append(f"lock_screen_orientation={self.config.lock_screen_orientation}")

        self.state.proc = subprocess.Popen(cmd)  # pylint: disable=consider-using-with

    def _stop_server(self) -> None:
        """Stop the scrcpy server."""
        if self.state.proc:
            self.state.proc.terminate()
            self.state.proc.wait()
        subprocess.run(self.adb_cmd + ["forward", "--remove", f"tcp:{self.config.port}"], check=True)
        if self.state.video_sock:
            self.state.video_sock.close()
            self.state.video_sock = None
        if self.state.control_sock:
            self.state.control_sock.close()
            self.state.control_sock = None

    def stop(self) -> None:
        """Stop the server and associated thread."""
        self._stop_server()
        if self.state.thread:
            self.state.thread.join()
        if self.state.control_thread:
            self.state.control_thread.join()

    def run(self) -> None:
        """Start the scrcpy client and video loop."""
        self._start_server()
        time.sleep(1)
        self.state.video_sock = socket.create_connection((self.config.host, self.config.port))
        if self.config.control:
            self.state.control_sock = socket.create_connection((self.config.host, self.config.port))
            read_exact(self.state.control_sock, 1)
            self.state.control_thread = threading.Thread(
                target=self._control_loop, args=(self.state.control_sock,), daemon=True
            )
            self.state.control_thread.start()

        self.state.thread = threading.Thread(
            target=self._video_loop, args=(self.state.video_sock,), daemon=True
        )
        self.state.thread.start()

    def _init_decoder(self, sock: socket.socket) -> Tuple[av.CodecContext, int, int]:
        """Initialize decoder and return decoder, width, and height."""
        _ = read_exact(sock, 1)
        self.state.device_name = read_exact(sock, 64).split(b"\0", 1)[0].decode()
        raw_codec = struct.unpack(">I", read_exact(sock, 4))[0]
        self.state.resolution = struct.unpack(">II", read_exact(sock, 8))
        width_, height_ = self.state.resolution
        codec_name = CODECS.get(raw_codec)
        if not codec_name:
            raise RuntimeError(f"Unsupported codec id: {raw_codec:#x}")
        print(f"Connected to '{self.state.device_name}': codec={codec_name} size={width_}x{height_}")
        decoder = av.CodecContext.create(codec_name, "r")
        return decoder, width_, height_

    def _video_loop(self, sock: socket.socket) -> None:
        """Main loop to receive and decode video packets."""
        try:
            decoder, _, _ = self._init_decoder(sock)
            config_data = b""

            while True:
                header = read_exact(sock, HEADER_SIZE)
                pts_flags, size = struct.unpack(">QI", header)
                packet_data = read_exact(sock, size)

                if pts_flags & FLAG_CONFIG:
                    config_data = packet_data
                    continue

                if config_data:
                    packet_data = config_data + packet_data
                    config_data = b""

                packet = av.Packet(packet_data)
                packet.pts = pts_flags & PTS_MASK
                if pts_flags & FLAG_KEY_FRAME:
                    try:
                        packet.is_keyframe = True
                    except AttributeError:
                        pass

                for decoded_frame in decoder.decode(packet):
                    img = decoded_frame.to_ndarray(format="rgb24")
                    self.state.last_frame = img

        finally:
            self._stop_server()

    def _control_loop(self, sock: socket.socket) -> None:
        """Receive device messages (e.g. clipboard)."""
        try:
            while True:
                msg_type = read_exact(sock, 1)[0]
                if msg_type == 0:  # DEVICE_MSG_TYPE_CLIPBOARD
                    length = struct.unpack(">I", read_exact(sock, 4))[0]
                    text = read_exact(sock, length).decode("utf-8")
                    print("Device clipboard:", text)
                elif msg_type == 1:  # DEVICE_MSG_TYPE_ACK_CLIPBOARD
                    _ = read_exact(sock, 8)
                elif msg_type == 2:  # DEVICE_MSG_TYPE_UHID_OUTPUT
                    id_, size = struct.unpack(">HH", read_exact(sock, 4))
                    _ = read_exact(sock, size)
                    print(f"UHID output id={id_} size={size}")
        except EOFError:
            pass

    def send_text(self, text: str) -> None:
        """Send an inject-text message."""
        if not self.state.control_sock:
            return
        raw = text.encode("utf-8")
        msg = struct.pack(">BI", 1, len(raw)) + raw
        self.state.control_sock.sendall(msg)

    def send_keycode(self, keycode: int, action: int, repeat: int = 0, meta: int = 0) -> None:
        """Send a keycode injection message."""
        if not self.state.control_sock:
            return
        msg = struct.pack(
            ">BBIII",
            0,  # TYPE_INJECT_KEYCODE
            action,
            keycode,
            repeat,
            meta,
        )
        self.state.control_sock.sendall(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scrcpy minimal client")
    parser.add_argument("--adb", default="adb", help="adb executable")
    parser.add_argument("--server", default="scrcpy-server-v3.3.1", help="path to scrcpy-server.jar")
    parser.add_argument("--host", default="127.0.0.1", help="host to connect to")
    parser.add_argument("--port", type=int, default=27183, help="local TCP port")
    parser.add_argument("--adb-host", default="127.0.0.1:5037", help="adb server host:port")
    parsed_args = parser.parse_args()

    config_obj = ClientConfig(
        adb=parsed_args.adb,
        server=parsed_args.server,
        host=parsed_args.host,
        port=parsed_args.port,
        ip=parsed_args.adb_host,
    )

    client = Client(config_obj)

    # GUI must be handled in main thread
    pygame.init()
    SCREEN = None
    clock = pygame.time.Clock()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                    keycode = ANDROID_KEYCODES.get(event.key)
                    if keycode is not None:
                        action = KEYEVENT_ACTION_DOWN if event.type == pygame.KEYDOWN else KEYEVENT_ACTION_UP
                        client.send_keycode(keycode, action)

            if client.state.last_frame is not None:
                current_frame = client.state.last_frame
                frame_height, frame_width, _ = current_frame.shape

                if SCREEN is None:
                    SCREEN = pygame.display.set_mode((frame_width, frame_height))

                surface = pygame.image.frombuffer(current_frame.tobytes(), (frame_width, frame_height), "RGB")
                SCREEN.blit(surface, (0, 0))
                pygame.display.flip()

            clock.tick(60)

    except KeyboardInterrupt:
        print("Exiting...")
        pygame.quit()
        client.stop()
