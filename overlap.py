import pickle
from numpy.core.overrides import set_module 
import torch
import numpy as np
from pcdet.utils import common_utils
import math

import open3d as o3d
import pykitti
import os
import matplotlib.pyplot as plt
from visual_utils import visualize_utils


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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
    # if these points are from bb then it would be NX7 otherwise NX3
    T_0_current = gt_poses[idx]
    T_prev_frame_0 = np.linalg.inv(gt_poses[idx - 1])
    
    if points.shape[1] == 7:
        xyz = points_to_homo(points[: , :3])
        points_w = T_prev_frame_0 @ T_0_current @ xyz.T
        points_w = np.delete(points_w.T, -1, axis=1)
        heading = np.copy(points[:, -1])

        R = (T_prev_frame_0 @ T_0_current)[:3, :3]
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            z = math.atan2(R[1,0], R[0,0])
        else :
            z = 0
        return np.hstack((points_w, points[:, 3:-1], (heading).reshape(-1,1)))
        
        
    if points.shape[1] == 3:
        xyz = points_to_homo(points)
        points_w = T_prev_frame_0 @ T_0_current @ xyz.T
        points[: , :3] = np.delete(points_w.T, -1, axis=1)
        return points

def get_gt_poses(data_path: str, sequence: str):
    
    """
    This function loads the GT poses on the Velodyne coordinate frame.
    """
    
    data = pykitti.odometry(data_path, sequence)
    T_cam_velo = data.calib.T_cam0_velo
    T_velo_cam = np.linalg.inv(T_cam_velo)
    gt_poses = T_velo_cam @ data.poses @ T_cam_velo
    return gt_poses

def points2pcd(points, intensities):
    pcd = o3d.geometry.PointCloud()
    colors = np.full_like(points, intensities.reshape(-1,1))

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def boxes_iou_normal(boxes_a, boxes_b):
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou
    

def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate
    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = np.abs(common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi))
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    Returns:
    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)

def custom_visualization(visual, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for i in visual:
        vis.add_geometry(i)
    vis.get_render_option().load_from_json("render_o3d.json")
    vis.run()
    vis.destroy_window()

def draw_3dbox(bbox_coords, color):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    This is how the corners are retured 
    Input: NX7 Bounding box corrds
    Output: Open3d Lineset object
    """
    visuals = []

    points = visualize_utils.boxes_to_corners_3d(bbox_coords)
    lines = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    
    if color == "red":
        colors = [[1, 0, 0] for i in range(len(lines))]
    if color == "something":
        colors = [[0, 1, 0] for i in range(len(lines))]
    for i in points:
        line_set = o3d.geometry.LineSet()
        line_set.points =  o3d.utility.Vector3dVector(i)
        line_set.lines =  o3d.utility.Vector2iVector(lines)
        line_set.colors =  o3d.utility.Vector3dVector(colors)
        visuals.append(line_set)
    return visuals

def cloud_n_detections(gt_poses, detections, idx, change_rf: bool, color):
    velo_path = list(detections.keys())[idx]

    points, intensities = bin2array(velo_path)

    boxes = detections[velo_path]['boxes'].numpy()
    if change_rf:
        points = cvt_frame(gt_poses, idx, points)
        boxes = cvt_frame(gt_poses, idx, boxes)

    pcd = points2pcd(points, intensities)
    o3d_box = draw_3dbox(boxes, color)

    return  [pcd] + o3d_box, points, boxes

def points_in_box(box, points: np.ndarray):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = visualize_utils.boxes_to_corners_3d(box).squeeze()

    p1 = corners[2]
    p_x = corners[3]
    p_y = corners[1]
    p_z = corners[6]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1
    
    iv = np.dot(v, i)
    jv = np.dot(v, j)
    kv = np.dot(v, k)


    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    return mask

def processing_consecutive_clouds(gt_poses, detections, coordinate, visual: bool):
    visual_1, points_1, boxes_1 = cloud_n_detections(gt_poses, detections, coordinate[0], change_rf=False, color="red")
    visual_2, points_2, boxes_2 = cloud_n_detections(gt_poses, detections, coordinate[1], change_rf=True, color="something")
    
    if visual: 
        custom_visualization(visual_1 + visual_2, window_name=f"Scan Number: {coordinate[0]} and {coordinate[1]} of Sequence 01")

    return [points_1, boxes_1], [points_2, boxes_2], [visual_1, visual_2]


def getting_point_mask(overlap_boxes, points):
    # print("Starting Masking of points inside the overlap box")
    segmented_idx = np.zeros(len(points), dtype=bool)
    pcd_masks = []
    for overlap_box in overlap_boxes:
        mask = points_in_box(overlap_box.reshape(1, -1), points)
        segmented_idx = segmented_idx + mask
        indices = mask.nonzero()[0]
        points_inside = points[indices]
        pcd_mask = points2pcd(points_inside, np.ones(len(points_inside)))
        pcd_mask.paint_uniform_color(np.array([1, 0, 0]))
        pcd_masks.append(pcd_mask)
    return pcd_masks, segmented_idx + 1

def saving_predictions(segmented_idx, path):
    fullname = os.path.basename(path)
    name, _ = os.path.splitext(fullname)
    sem_label = segmented_idx.astype(np.uint16)
    inst_label = np.zeros_like(sem_label, dtype=np.uint32)
    label = sem_label + inst_label
    label.tofile(f"predictions/{name}.label")
    return 


def loading_predictions(path):
    label = np.fromfile(path, dtype=np.uint32)
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF
    return label

def consecutive_idx_pairs(len_detections):
    idxx = np.arange(len_detections)
    coordinates = np.vstack((idxx, idxx+1)).T[:-1]
    return coordinates

def main():

    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    detections_path = "obj_boxes.pkl"
    sequence = "08"

    gt_poses = get_gt_poses(data_path, sequence)
    
    #detections loaded  
    detections = load_pickle(detections_path)
    coordinates = consecutive_idx_pairs(len(detections))

    for i in coordinates:
        pts_bb_1 , pts_bb_2, visuals_12 =  processing_consecutive_clouds(gt_poses, detections, i, visual=False)
        
        iou = boxes3d_nearest_bev_iou(torch.tensor(pts_bb_1[1]), torch.tensor(pts_bb_2[1]))

        # get box coordinates with iou higher than 40
        static_coords = (iou > 0.8).nonzero().T[0]     # static objects
        moving_coords = np.setxor1d(np.arange(len(pts_bb_1[1])), static_coords)
        
        if len(moving_coords) == 0:
            print(f"No boxes found for scan {i[0]}")

        overlap_boxes = pts_bb_1[1][moving_coords].reshape(-1, 7)
        
        pcd_masks, segmented_idx = getting_point_mask(overlap_boxes, pts_bb_1[0])
        # saving_predictions(segmented_idx, list(detections.keys())[i[0]])
        custom_visualization(visuals_12[0] + visuals_12[1] + pcd_masks , window_name=f"Scan Number: {i[0]} and {i[1]} of Sequence 08")

        # input()
        
    # raise NameError('HiThere')

if __name__ == "__main__":
    main()

