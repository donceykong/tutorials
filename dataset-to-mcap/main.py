# Written by Doncey Albin

# External imports
import os
import struct
from io import BytesIO
from random import random

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.Color_pb2 import Color

# Foxglove Imports
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer
from tqdm import tqdm

# Internal imports
from utils.kitti360 import KITTI360


def seconds_to_nanoseconds(seconds):
    return int(seconds * 1000000000)


def minutes_to_nanoseconds(minutes):
    seconds = 60 * minutes
    return seconds_to_nanoseconds(seconds)


def tf_matrix_to_quaternion(tf_matrix):
    x, y, z, _ = tf_matrix[:, 3]
    R = tf_matrix[0:3, 0:3]

    # rotation matrix (RPY)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    # RPY to quaternion
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)

    return x, y, z, qw, qx, qy, qz


def convert_to_mcap(kitti360):
    # frame_points_accum = []
    # rgb_np_accum = []
    with open(f"./kitti360_seq_{kitti360.seq}.mcap", "wb") as f, Writer(f) as writer:
        for frame_number, data_tag in tqdm(
            list(kitti360.data_timestamps.keys()), desc="Processing timestamps"
        ):
            timestamp_tuple = kitti360.data_timestamps[(frame_number, data_tag)]

            velo_to_map_tf_matrix = kitti360.velodyne_poses.get(frame_number)
            write_tf_data(writer, timestamp_tuple, velo_to_map_tf_matrix)

            if data_tag == "lidar":
                points = kitti360.get_velo_points(frame_number)
                labels_np = kitti360.get_velo_labels(frame_number)
                points_rgb = kitti360.color_point_cloud(
                    points, labels_np, kitti360.labels_dict
                )
                write_velo_to_mcap(
                    points,
                    points_rgb,
                    writer,
                    timestamp_tuple,
                    frame_id="velodyne_frame",
                    topic="/velodyne_pointcloud",
                )

                # ###############
                # # ACCUM PONTS
                # ###############
                # # TF Points
                # pose = kitti360.velodyne_poses.get(frame_number)
                # transformed_points = kitti360.get_transformed_point_cloud(points, pose)
                # downsampled_tf_points, downsampled_rgb = kitti360.downsample_pointcloud(
                #     transformed_points, points_rgb, voxel_leafsize=0.25
                # )

                # # Append points and colors to the lists
                # frame_points_accum.append(downsampled_tf_points)
                # rgb_np_accum.append(downsampled_rgb)

                # # Concatenate all accumulated points and colors
                # if frame_points_accum:
                #     concat_points = np.concatenate(frame_points_accum, axis=0)
                #     concat_rgb = np.concatenate(rgb_np_accum, axis=0)

                #     # Downsample points
                #     downsampled_points_accum, downsampled_rgb_accum = (
                #         kitti360.downsample_pointcloud(
                #             concat_points,
                #             concat_rgb,
                #             voxel_leafsize=0.5,
                #         )
                #     )

                #     # Write scan to mcap file
                #     write_velo_to_mcap(
                #         downsampled_points_accum,
                #         downsampled_rgb_accum,
                #         writer,
                #         timestamp_tuple,
                #         frame_id="map",
                #         topic="/velodyne_accum",
                #     )

            # elif data_tag == "gps":
            #     pose_lla = kitti360.velodyne_poses_latlon.get(frame_number)[:3]
            #     gps_frame = "velodyne_gps_frame"
            #     write_gps_to_mcap(writer, timestamp_tuple, pose_lla, gps_frame)
            elif data_tag == "cam_00":
                raw_frame_img, semantic_frame_img = kitti360.get_cam_images(
                    frame_number, camera_number=0
                )
                # # Write scan to mcap file
                write_cam_to_mcap(
                    writer,
                    raw_frame_img,
                    timestamp_tuple,
                    topic="/camera_00",
                    frame_id="camera_00_frame",
                )
                # # Write scan to mcap file
                write_cam_to_mcap(
                    writer,
                    semantic_frame_img,
                    timestamp_tuple,
                    topic="/camera_00_semantic",
                    frame_id="camera_00_frame",
                )
            elif data_tag == "cam_01":
                raw_frame_img, semantic_frame_img = kitti360.get_cam_images(
                    frame_number, camera_number=1
                )
                # # Write scan to mcap file
                write_cam_to_mcap(
                    writer,
                    raw_frame_img,
                    timestamp_tuple,
                    topic="/camera_01",
                    frame_id="camera_01_frame",
                )
                # # Write scan to mcap file
                write_cam_to_mcap(
                    writer,
                    semantic_frame_img,
                    timestamp_tuple,
                    topic="/camera_01_semantic",
                    frame_id="camera_01_frame",
                )


def write_tf_data(writer, timestamp_tuple, velo_to_map_tf):
    write_tf_to_mcap(
        writer,
        timestamp_tuple,
        velo_to_map_tf,
        parent_frame="map",
        child_frame="velodyne_frame",
    )

    camera_00_to_velo_tf = np.array(
        [[0, 0, 1, 0.79], [-1, 0, 0, 0.3], [0, -1, 0, -0.18], [0, 0, 0, 1]]
    )
    parent_frame = "velodyne_frame"
    child_frame = "camera_00_frame"
    write_tf_to_mcap(
        writer, timestamp_tuple, camera_00_to_velo_tf, parent_frame, child_frame
    )

    camera_01_to_velo_tf = np.array(
        [[0, 0, 1, 0.79], [-1, 0, 0, -0.3], [0, -1, 0, -0.18], [0, 0, 0, 1]]
    )
    parent_frame = "velodyne_frame"
    child_frame = "camera_01_frame"
    write_tf_to_mcap(
        writer, timestamp_tuple, camera_01_to_velo_tf, parent_frame, child_frame
    )


def write_velo_to_mcap(points, points_rgb, writer, timestamp_tuple, frame_id, topic):
    fields = [
        PackedElementField(name="x", offset=0, type=PackedElementField.FLOAT32),
        PackedElementField(name="y", offset=4, type=PackedElementField.FLOAT32),
        PackedElementField(name="z", offset=8, type=PackedElementField.FLOAT32),
        PackedElementField(
            name="intensity", offset=12, type=PackedElementField.FLOAT32
        ),
        PackedElementField(name="red", offset=16, type=PackedElementField.UINT8),
        PackedElementField(name="green", offset=17, type=PackedElementField.UINT8),
        PackedElementField(name="blue", offset=18, type=PackedElementField.UINT8),
        PackedElementField(name="alpha", offset=19, type=PackedElementField.UINT8),
    ]

    pose_msg = Pose(
        position=Vector3(x=0, y=0, z=0),
        orientation=Quaternion(w=0, x=0, y=0, z=0),
    )

    data = BytesIO()
    for point, point_rgb in zip(points, points_rgb):
        x = np.float32(point[0])
        y = np.float32(point[1])
        z = np.float32(point[2])
        intensity = np.float32(point[3])
        red = np.uint8(point_rgb[0])
        green = np.uint8(point_rgb[1])
        blue = np.uint8(point_rgb[2])
        alpha = np.uint8(255)

        data.write(
            struct.pack(
                "ffffBBBB",  # Note 'B' for unsigned 8-bit integer and 'f' for 32-bit float
                x,
                y,
                z,
                intensity,
                red,
                green,
                blue,
                alpha,
            )
        )

    nanoseconds, epoch_seconds = timestamp_tuple
    point_stride = (
        20  # 4 float fields (4 bytes each) and four UINT8 fields (1 bytes each)
    )
    pc_msg = PointCloud(
        frame_id=frame_id,
        pose=pose_msg,
        timestamp=Timestamp(seconds=epoch_seconds, nanos=nanoseconds),
        point_stride=point_stride,
        fields=fields,
        data=data.getvalue(),
    )
    writer.write_message(
        topic=topic,
        log_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
        message=pc_msg,
        publish_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
    )


def write_cam_to_mcap(writer, frame_img, timestamp_tuple, topic, frame_id):
    width = frame_img.shape[1]
    height = frame_img.shape[0]

    data = BytesIO()
    # Pack images RGB info
    for y in range(height):
        for x in range(width):
            r = frame_img[y, x, 0]
            g = frame_img[y, x, 1]
            b = frame_img[y, x, 2]
            data.write(struct.pack("BBB", r, g, b))

    # /camera/image
    nanoseconds, epoch_seconds = timestamp_tuple
    img = RawImage(
        timestamp=Timestamp(seconds=epoch_seconds, nanos=nanoseconds),
        frame_id=frame_id,
        width=width,
        height=height,
        encoding="rgb8",
        step=width * 3,
        data=data.getvalue(),
    )
    writer.write_message(
        topic=f"{topic}/image",
        log_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
        message=img,
        publish_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
    )

    # Camera parameters for /camera_00 and /camera_01
    camera_params = {
        "/camera_00": {
            "K": [
                788.629315,
                0.0,
                687.158398,
                0.0,
                786.382230,
                317.752196,
                0.0,
                0.0,
                1.0,
            ],
            "D": [-0.344441, 0.141678, 0.000414, -0.000222, -0.029608],
            "R_rect": [
                0.999974,
                -0.007141,
                -0.000089,
                0.007141,
                0.999969,
                -0.003247,
                0.000112,
                0.003247,
                0.999995,
            ],
            "P_rect": [
                552.554261,
                0.0,
                682.049453,
                0.0,
                0.0,
                552.554261,
                238.769549,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            "S": [1408, 376],
        },
        "/camera_00_semantic": {
            "K": [
                788.629315,
                0.0,
                687.158398,
                0.0,
                786.382230,
                317.752196,
                0.0,
                0.0,
                0.0,
            ],
            "D": [-0.344441, 0.141678, 0.000414, -0.000222, -0.029608],
            "R_rect": [
                0.999974,
                -0.007141,
                -0.000089,
                0.007141,
                0.999969,
                -0.003247,
                0.000112,
                0.003247,
                0.999995,
            ],
            "P_rect": [
                552.554261,
                0.0,
                682.049453,
                0.0,
                0.0,
                552.554261,
                238.769549,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            "S": [1408, 376],
        },
        "/camera_01": {
            "K": [
                785.134093,
                0.0,
                686.437073,
                0.0,
                782.346289,
                321.352788,
                0.0,
                0.0,
                0.0,
            ],
            "D": [-0.353195, 0.161996, 0.000383, -0.000242, -0.041476],
            "R_rect": [
                0.999837,
                0.004862,
                -0.017390,
                -0.004974,
                0.999967,
                -0.006389,
                0.017358,
                0.006474,
                0.999828,
            ],
            "P_rect": [
                552.554261,
                0.0,
                682.049453,
                -328.318735,
                0.0,
                552.554261,
                238.769549,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            "S": [1408, 376],
        },
        "/camera_01_semantic": {
            "K": [
                785.134093,
                0.0,
                686.437073,
                0.0,
                782.346289,
                321.352788,
                0.0,
                0.0,
                0.0,
            ],
            "D": [-0.353195, 0.161996, 0.000383, -0.000242, -0.041476],
            "R_rect": [
                0.999837,
                0.004862,
                -0.017390,
                -0.004974,
                0.999967,
                -0.006389,
                0.017358,
                0.006474,
                0.999828,
            ],
            "P_rect": [
                552.554261,
                0.0,
                682.049453,
                -328.318735,
                0.0,
                552.554261,
                238.769549,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            "S": [1408, 376],
        },
    }

    if topic in camera_params:
        params = camera_params["/camera_00"]
        K = params["K"]
        D = params["D"]
        R_rect = params["R_rect"]
        P_rect = params["P_rect"]
        S = params["S"]
    else:
        # Handle unknown topic
        raise ValueError(f"Unknown topic: {topic}")

    width, height = int(S[0]), int(S[1])

    # /camera/calibration
    cal = CameraCalibration(
        timestamp=Timestamp(seconds=epoch_seconds, nanos=nanoseconds),
        frame_id=frame_id,
        width=width,
        height=height,
        distortion_model="plumb_bob",
        D=D,
        K=K,
        R=R_rect,
        P=P_rect,
    )
    writer.write_message(
        topic=f"{topic}/calibration",
        log_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
        message=cal,
        publish_time=(nanoseconds + seconds_to_nanoseconds(epoch_seconds)),
    )


def write_tf_to_mcap(writer, timestamp_tuple, tf_matrix, parent_frame, child_frame):
    x, y, z, qw, qx, qy, qz = tf_matrix_to_quaternion(tf_matrix)
    nanoseconds, epoch_seconds = timestamp_tuple

    # Create and publish TF Message
    tf_msg = FrameTransform(
        timestamp=Timestamp(seconds=epoch_seconds, nanos=nanoseconds),
        parent_frame_id=parent_frame,
        child_frame_id=child_frame,
        translation=Vector3(x=x, y=y, z=z),
        rotation=Quaternion(w=qw, x=qx, y=qy, z=qz),
    )
    writer.write_message(
        topic="/tf",
        log_time=int((nanoseconds + seconds_to_nanoseconds(epoch_seconds))),
        message=tf_msg,
        publish_time=int((nanoseconds + seconds_to_nanoseconds(epoch_seconds))),
    )


def write_gps_to_mcap(writer, timestamp_tuple, pose_lla, gps_frame):
    lat = pose_lla[1][3]
    lon = pose_lla[0][3]
    alt = pose_lla[2][3]

    nanoseconds, epoch_seconds = timestamp_tuple

    # Create and publish TF Message
    gps_msg = LocationFix(
        timestamp=Timestamp(seconds=epoch_seconds, nanos=nanoseconds),
        frame_id=gps_frame,
        latitude=lat,
        longitude=lon,
        altitude=alt,
    )
    writer.write_message(
        topic="/LocationFix",
        log_time=int((nanoseconds + seconds_to_nanoseconds(epoch_seconds))),
        message=gps_msg,
        publish_time=int((nanoseconds + seconds_to_nanoseconds(epoch_seconds))),
    )


def main():
    sequence = 0
    minute_begin = 5
    minute_end = 5.1
    nanosec_begin = minutes_to_nanoseconds(minute_begin)
    nanosec_end = minutes_to_nanoseconds(minute_end)
    kitti360 = KITTI360(sequence, nanosec_begin, nanosec_end)

    print(f"\nConverting sequence {sequence} data to mcap")
    convert_to_mcap(kitti360)


if __name__ == "__main__":
    main()
