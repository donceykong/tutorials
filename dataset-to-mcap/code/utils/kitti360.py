import glob
import os

import numpy as np
import open3d as o3d
from PIL import Image
from utils.labels import labels


class FileOperations:
    """Base class for file operations."""

    def find_min_max_file_names(self, file_dir_path, number_delimiter, file_extension):
        pattern = os.path.join(file_dir_path, f"*{file_extension}")
        files = glob.glob(pattern)
        if number_delimiter is not None:
            file_numbers = [
                int(os.path.basename(file).split(f"{number_delimiter}")[0])
                for file in files
            ]
        else:
            file_numbers = [int(os.path.basename(file).split(".")[0]) for file in files]
        return min(file_numbers), max(file_numbers) if file_numbers else (None, None)

    def save_accum_points_and_rgb(
        self, accum_points_dir, frame_num, accum_points, accum_rgb
    ):
        accum_points_file = os.path.join(
            accum_points_dir, f"{frame_num:010d}_accum_points"
        )
        accum_rgb_file = os.path.join(accum_points_dir, f"{frame_num:010d}_accum_rgb")
        np.save(accum_points_file, accum_points)
        np.save(accum_rgb_file, accum_rgb)

    def get_accum_points_and_rgb(self, accum_points_file, accum_rgb_file):
        accum_points = np.load(accum_points_file, allow_pickle=True)
        accum_rgb = np.load(accum_rgb_file, allow_pickle=True)
        return accum_points, accum_rgb

    def save_poses(self, file_path, xyz_poses):
        with open(file_path, "w") as file:
            for idx, matrix_4x4 in xyz_poses.items():
                flattened_matrix = matrix_4x4.flatten()
                line = f"{idx} " + " ".join(map(str, flattened_matrix)) + "\n"
                file.write(line)

    def read_poses(self, file_path):
        poses_xyz = {}
        with open(file_path, "r") as file:
            for line in file:
                elements = line.strip().split()
                frame_index = int(elements[0])
                if len(elements[1:]) == 16:
                    matrix_4x4 = np.array(elements[1:], dtype=float).reshape((4, 4))
                else:
                    matrix_3x4 = np.array(elements[1:], dtype=float).reshape((3, 4))
                    matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
                poses_xyz[frame_index] = matrix_4x4
        return poses_xyz

    def read_png_file(self, png_file):
        img = Image.open(png_file)
        return np.array(img)


class PointCloudOperations:
    """Base class for point cloud operations."""

    def get_transformed_point_cloud(self, pc, transformation_matrix):
        xyz = pc[:, :3]
        if pc.shape[1] > 3:
            intensity = pc[:, 3].reshape(-1, 1)
        xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        transformed_xyz = np.dot(xyz_homogeneous, transformation_matrix.T)[:, :3]
        if pc.shape[1] > 3:
            transformed_points = np.concatenate([transformed_xyz, intensity], axis=1)
        else:
            transformed_points = transformed_xyz
        return transformed_points

    def color_point_cloud(self, points, labels, labels_dict):
        colored_points = np.zeros_like(points[:, :3])
        for i, label in enumerate(labels):
            if np.isnan(label) or label == -1:
                continue
            color = labels_dict.get(label, (0, 0, 0))
            colored_points[i] = np.array(color)
        return colored_points

    def downsample_pointcloud(self, pointcloud_accum, rgb_accum, voxel_leafsize):
        pointcloud_xyz = pointcloud_accum[:, :3]
        pointcloud_intensities = pointcloud_accum[:, 3]
        monochrome_colors = np.stack([pointcloud_intensities] * 3, axis=-1)
        colors_pcd = o3d.geometry.PointCloud()
        colors_pcd.points = o3d.utility.Vector3dVector(pointcloud_xyz)
        colors_pcd.colors = o3d.utility.Vector3dVector(rgb_accum)
        downsampled_colors = colors_pcd.voxel_down_sample(voxel_size=voxel_leafsize)
        downsampled_rgb_accum = np.asarray(downsampled_colors.colors)
        intensities_pcd = o3d.geometry.PointCloud()
        intensities_pcd.points = o3d.utility.Vector3dVector(pointcloud_xyz)
        intensities_pcd.colors = o3d.utility.Vector3dVector(monochrome_colors)
        downsampled_intensities = intensities_pcd.voxel_down_sample(
            voxel_size=voxel_leafsize
        )
        downsampled_intensities_accum = np.asarray(downsampled_intensities.colors)[:, 0]
        downsampled_points_xyz = np.asarray(downsampled_intensities.points)
        downsampled_points_accum = np.concatenate(
            [downsampled_points_xyz, downsampled_intensities_accum[:, np.newaxis]],
            axis=1,
        )
        return downsampled_points_accum, downsampled_rgb_accum

    def get_poses_lidar_from_imu(self, imu_poses_xyz):
        translation_vector = np.array([0.81, 0.32, -0.83])
        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        imu_to_lidar_matrix = np.identity(4)
        imu_to_lidar_matrix[:3, :3] = rotation_matrix
        imu_to_lidar_matrix[:3, 3] = translation_vector
        lidar_poses_xyz = {}
        for idx, imu_matrix in imu_poses_xyz.items():
            lidar_pose = np.matmul(imu_matrix, imu_to_lidar_matrix)
            lidar_poses_xyz[idx] = lidar_pose
        return lidar_poses_xyz


import datetime


def seconds_to_nanoseconds(seconds):
    return int(seconds * 1000000000)


def insert_timestamp_data(sensor_timestamp_dict, timestamp_path, data_tag):
    """Reads the timestamp file and returns a list of (seconds, nanoseconds) tuples."""
    with open(timestamp_path, "r") as file:
        for frame_number, line in enumerate(file):
            timestamp_str = line.strip()
            # Split date and time
            date_part, time_part = timestamp_str.split(" ")
            # Further split time into seconds and deci-seconds
            seconds_str, fractional_seconds_str = (
                time_part.split(".")[0],
                float("." + time_part.split(".")[1]),
            )
            # Combine date and seconds part
            full_datetime_str = f"{date_part} {seconds_str}"
            # Convert to datetime object
            datetime_obj = datetime.datetime.strptime(
                full_datetime_str, "%Y-%m-%d %H:%M:%S"
            )
            # Convert to epoch seconds and ensure is casted to int
            epoch_seconds = int(datetime_obj.timestamp())
            # Convert nanoseconds part
            nanoseconds = seconds_to_nanoseconds(fractional_seconds_str)
            # Append tuple to the list
            sensor_timestamp_dict[frame_number, data_tag] = [nanoseconds, epoch_seconds]
    return sensor_timestamp_dict


class KITTI360(FileOperations, PointCloudOperations):
    def __init__(self, seq, nanosec_begin, nanosec_end):
        self.seq = seq
        self.labels_dict = {label.id: label.color for label in labels}
        self.setup_path_variables()

        self.set_timestamps(nanosec_begin, nanosec_end)
        self.filter_timestamps()
        self.get_velo_poses()

    def set_timestamps(self, nanosec_begin, nanosec_end):
        self.data_timestamps = {}
        insert_timestamp_data(self.data_timestamps, self.velo_timestamp_file, "lidar")
        insert_timestamp_data(self.data_timestamps, self.cam_00_timestamps, "cam_00")
        insert_timestamp_data(self.data_timestamps, self.cam_01_timestamps, "cam_01")
        insert_timestamp_data(self.data_timestamps, self.gps_timestamps, "gps")

        earliest_timestamp = np.inf
        for nanoseconds, epoch_seconds in self.data_timestamps.values():
            current_timestamp = nanoseconds + seconds_to_nanoseconds(epoch_seconds)
            if current_timestamp < earliest_timestamp:
                earliest_timestamp = current_timestamp

        self.timestamp_min = nanosec_begin + earliest_timestamp
        self.timestamp_max = nanosec_end + earliest_timestamp

    def filter_timestamps(self):
        """
        Only use velodyne timestamps for frames where lidar labels exist
        Only use camera timestamps for frames where camera labels exist
        """
        filtered_data_timestamps = self.data_timestamps.copy()
        for frame_number, data_tag in self.data_timestamps.keys():
            nanoseconds, epoch_seconds = self.data_timestamps[(frame_number, data_tag)]
            current_timestamp = nanoseconds + seconds_to_nanoseconds(epoch_seconds)

            velo_label_file = os.path.join(self.label_path, f"{frame_number:010d}.bin")
            cam_00_semantic_img_file = os.path.join(
                self.cam_00_semantic_images_path, f"{frame_number:010d}.png"
            )
            cam_01_semantic_img_file = os.path.join(
                self.cam_01_semantic_images_path, f"{frame_number:010d}.png"
            )
            gps_pose_file = os.path.join(
                self.gps_poses_path, f"{frame_number:010d}.txt"
            )

            semantics_file = None
            if data_tag == str("lidar"):
                semantics_file = velo_label_file
            elif data_tag == str("cam_00"):
                semantics_file = cam_00_semantic_img_file
            elif data_tag == str("cam_01"):
                semantics_file = cam_01_semantic_img_file
            elif data_tag == str("gps"):
                semantics_file = gps_pose_file

            if (
                not os.path.exists(semantics_file)
                or current_timestamp < self.timestamp_min
                or current_timestamp > self.timestamp_max
            ):
                filtered_data_timestamps.pop((frame_number, data_tag), None)
        self.data_timestamps = filtered_data_timestamps.copy()

    def get_cam_images(self, frame, camera_number):
        cam_frame_path = None
        cam_semantic_frame_path = None
        if camera_number == 0:
            cam_frame_path = os.path.join(
                self.cam_00_raw_images_path, f"{frame:010d}.png"
            )
            cam_semantic_frame_path = os.path.join(
                self.cam_00_semantic_images_path, f"{frame:010d}.png"
            )
        elif camera_number == 1:
            cam_frame_path = os.path.join(
                self.cam_01_raw_images_path, f"{frame:010d}.png"
            )
            cam_semantic_frame_path = os.path.join(
                self.cam_01_semantic_images_path, f"{frame:010d}.png"
            )
        raw_frame_img = self.read_png_file(cam_frame_path)
        semantic_frame_img = self.read_png_file(cam_semantic_frame_path)
        return raw_frame_img, semantic_frame_img

    def get_velo_poses(self):
        if os.path.exists(self.velodyne_poses_file):
            self.velodyne_poses = self.read_poses(self.velodyne_poses_file)
        else:
            imu_poses_xyz = self.read_poses(self.imu_poses_file)
            self.velodyne_poses = self.get_poses_lidar_from_imu(imu_poses_xyz)
            self.save_poses(self.velodyne_poses_file, self.velodyne_poses)

    def get_gps_pose(self, frame):
        gps_pose_file = os.path.join(self.gps_poses_path, f"{frame:010d}.txt")
        with open(gps_pose_file, "r") as file:
            first_line = file.readline().strip()
            lat_lon_alt = np.array(first_line.split()[:3], dtype=np.float32)
            return lat_lon_alt

    def get_velo_points(self, frame):
        raw_pc_frame_path = os.path.join(self.raw_pc_path, f"{frame:010d}.bin")
        point_cloud = np.fromfile(raw_pc_frame_path, dtype=np.float32)
        return point_cloud.reshape(-1, 4)

    def get_velo_labels(self, frame):
        velo_label_path = os.path.join(self.label_path, f"{frame:010d}.bin")
        labels = np.fromfile(velo_label_path, dtype=np.int16)
        return labels.reshape(-1)

    def setup_path_variables(self):
        self.kitti360Path = os.getenv(
            "KITTI360_DATASET",
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..", "data/KITTI-360"
            ),
        )
        sequence_dir = f"2013_05_28_drive_{self.seq:04d}_sync"
        self.raw_pc_path = os.path.join(
            self.kitti360Path, "data_3d_raw", sequence_dir, "velodyne_points", "data"
        )
        self.velo_timestamp_file = os.path.join(
            self.kitti360Path,
            "data_3d_raw",
            sequence_dir,
            "velodyne_points",
            "timestamps.txt",
        )
        self.semantics_dir_path = os.path.join(
            self.kitti360Path, "data_3d_semantics", sequence_dir
        )
        self.accum_points_dir = os.path.join(
            self.kitti360Path, "data_3d_accum", sequence_dir
        )
        self.label_path = os.path.join(self.semantics_dir_path, "labels")
        self.accum_ply_path = os.path.join(self.semantics_dir_path, "accum_ply")
        self.imu_poses_file = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "poses.txt"
        )
        self.velodyne_poses_file = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "velodyne_poses.txt"
        )
        self.oxts_pose_file_path = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "poses_latlong.txt"
        )
        self.extracted_per_frame_dir = os.path.join(
            self.kitti360Path, "data_3d_extracted", sequence_dir, "per_frame"
        )
        self.osm_file_path = os.path.join(
            self.kitti360Path, "data_osm", f"map_{self.seq:04d}.osm"
        )
        self.cam_00_timestamps = os.path.join(
            self.kitti360Path, "data_2d_raw", sequence_dir, "image_00/timestamps.txt"
        )
        self.cam_00_raw_images_path = os.path.join(
            self.kitti360Path, "data_2d_raw", sequence_dir, "image_00/data_rect/"
        )
        self.cam_00_semantic_images_path = os.path.join(
            self.kitti360Path,
            "data_2d_semantics/train",
            sequence_dir,
            "image_00/semantic_rgb/",
        )
        self.cam_01_timestamps = os.path.join(
            self.kitti360Path, "data_2d_raw", sequence_dir, "image_01/timestamps.txt"
        )
        self.cam_01_raw_images_path = os.path.join(
            self.kitti360Path, "data_2d_raw", sequence_dir, "image_01/data_rect/"
        )
        self.cam_01_semantic_images_path = os.path.join(
            self.kitti360Path,
            "data_2d_semantics/train",
            sequence_dir,
            "image_01/semantic_rgb",
        )

        self.gps_poses_path = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "oxts/data/"
        )
        self.gps_timestamps = os.path.join(
            self.kitti360Path, "data_poses", sequence_dir, "oxts/timestamps.txt"
        )
