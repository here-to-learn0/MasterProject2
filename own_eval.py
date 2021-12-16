import open3d as o3d
import numpy as np
import pykitti
import os
import matplotlib.pyplot as plt


def bin2array(file):
    array = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    points = array[:, :3]
    intensity = array[:, -1]
    return points, intensity

def points_to_homo(points):
    ones = np.ones(len(points)).reshape(-1,1)
    points = np.hstack((points, ones))
    return points

def cvt_frame(gt_poses, idx, points):
    points_w = gt_poses[idx] @ points.T
    points_w = np.delete(points_w.T, -1, axis=1)
    return points_w

def get_gt_poses():
    """This function loads the GT poses on the Velodyne coordinate frame."""
    data = pykitti.odometry("/media/sandhu/SSD/dataset/kitti/odometry/dataset", "08")
    T_cam_velo = data.calib.T_cam0_velo
    T_velo_cam = np.linalg.inv(T_cam_velo)

    gt_poses = T_velo_cam @ data.poses @ T_cam_velo
    return gt_poses

def points2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


if __name__ == "__main__":
    path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset/sequences/08/velodyne/"
    bin_file_path = sorted(os.listdir(path))
    full_file_path = [os.path.join(path, i) for i in bin_file_path]

    gt_poses = get_gt_poses()

    for idx, i in enumerate(full_file_path):
        points, _ = bin2array(i)
        points = points_to_homo(points)
        points = cvt_frame(gt_poses, idx, points)
        pcd = points2pcd(points)
        o3d.io.write_point_cloud(f"pcd/{idx}.pcd", pcd)