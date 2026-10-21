import numpy as np


def depth_to_point_map(depth: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray):
    H, W = depth.shape
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    y, x = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing="ij")

    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)

    return points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
