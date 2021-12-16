import numpy as np
import torch
import overlap


def getting_point_mask(boxes, points, labels):
    print("Starting Masking of points inside the overlap box")
    segmented_idx = np.zeros(len(points), dtype=bool)
    boxes = np.array(boxes)
    for i, box in enumerate(boxes):

        mask = overlap.points_in_box(box.reshape(1, -1), points) * labels[i].item()
        segmented_idx = segmented_idx + mask
    return segmented_idx


def main():
    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    detection_path = "obj_boxes.pkl"
    detections = overlap.load_pickle(detection_path)
    
    for idx, velo_path in enumerate(detections):
        _, points, _ = overlap.cloud_n_detections(None, detections, idx, False, "red")
        
        seg_labels = getting_point_mask(detections[velo_path]['boxes'], points, detections[velo_path]['labels'])
        overlap.saving_predictions(seg_labels, velo_path)


if __name__ == "__main__":
    main()

