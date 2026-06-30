import os
import struct
import zlib

import cv2
import imageio.v2 as imageio
import numpy as np


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            array = np.frombuffer(self.color_data, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode JPEG color frame")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if compression_type == "png":
            return imageio.imread(self.color_data)
        raise ValueError(f"Unsupported color compression: {compression_type}")


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as handle:
            version = struct.unpack("I", handle.read(4))[0]
            if version != self.version:
                raise ValueError(f"Unsupported SensorData version: {version}")
            strlen = struct.unpack("Q", handle.read(8))[0]
            self.sensor_name = handle.read(strlen).decode("utf-8", errors="replace")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", handle.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", handle.read(4))[0]]
            self.color_width = struct.unpack("I", handle.read(4))[0]
            self.color_height = struct.unpack("I", handle.read(4))[0]
            self.depth_width = struct.unpack("I", handle.read(4))[0]
            self.depth_height = struct.unpack("I", handle.read(4))[0]
            self.depth_shift = struct.unpack("f", handle.read(4))[0]
            num_frames = struct.unpack("Q", handle.read(8))[0]
            self.frames = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames) // frame_skip} depth frames to {output_path}")
        for frame_idx in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[frame_idx].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imwrite(os.path.join(output_path, f"{frame_idx}.png"), depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames) // frame_skip} color frames to {output_path}")
        for frame_idx in range(0, len(self.frames), frame_skip):
            color = self.frames[frame_idx].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, f"{frame_idx}.jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as handle:
            for line in matrix:
                np.savetxt(handle, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames) // frame_skip} camera poses to {output_path}")
        for frame_idx in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[frame_idx].camera_to_world,
                os.path.join(output_path, f"{frame_idx}.txt"),
            )

    def export_intrinsics(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting camera intrinsics to {output_path}")
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt"))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt"))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt"))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt"))
