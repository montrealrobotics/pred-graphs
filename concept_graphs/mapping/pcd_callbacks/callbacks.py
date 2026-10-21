import open3d as o3d
from .PointCloudCallback import PointCloudCallback
import numpy as np


class DBSCAN(PointCloudCallback):
    def __init__(self, eps=0.02, min_points=10):
        self.eps = eps
        self.min_points = min_points

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) < self.min_points:
            return pcd

        labels = pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points)
        labels_unique, counts = np.unique(labels, return_counts=True)
        max_label = labels_unique[np.argmax(counts)]
        mask = labels == max_label

        return pcd.select_by_index(np.where(mask)[0])


class VoxelDownSampling(PointCloudCallback):
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(self.voxel_size)
