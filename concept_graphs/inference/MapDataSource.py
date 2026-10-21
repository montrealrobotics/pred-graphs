import json
import glob
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
import open3d as o3d
from natsort import natsorted

from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.utils import load_point_cloud

log = logging.getLogger(__name__)


class MapDataSource(ABC):
    """
    Abstract interface for map data access.

    This abstraction allows MapQueryEngine to work with different data sources:
    - Disk-based maps (exported ObjectMaps with segments_anno.json, features, etc.)
    - In-memory ObjectMaps (for real-time processing without disk I/O)
    """

    @property
    @abstractmethod
    def features(self) -> torch.Tensor:
        """CLIP features for all objects in the map. Shape: (N, D)."""
        ...

    @property
    @abstractmethod
    def bbox(self) -> List[o3d.geometry.OrientedBoundingBox]:
        """Oriented bounding boxes for all objects."""
        ...

    @property
    @abstractmethod
    def annotations(self) -> List[Dict]:
        """Annotation data for each object (id, centroid, timestamps, etc.)."""
        ...

    @abstractmethod
    def get_object_images(self, object_id: int, limit: int = 5) -> List[np.ndarray]:
        """Get RGB images for a specific object.

        Args:
            object_id: Index of the object in the map
            limit: Maximum number of images to return

        Returns:
            List of RGB images as numpy arrays
        """
        ...

    @abstractmethod
    def get_object_centroid(self, object_id: int) -> np.ndarray:
        """Get centroid position of a specific object.

        Args:
            object_id: Index of the object in the map

        Returns:
            3D centroid position as numpy array
        """
        ...

    def get_object_labels(self, object_id: int) -> List[str]:
        """Get the list of segment labels for a specific object.

        This method returns the labels from all segments of the object,
        which may include duplicates if multiple segments have the same label.

        Args:
            object_id: Index of the object in the map

        Returns:
            List of label strings from all segments
        """
        ...


class DiskMapDataSource(MapDataSource):
    """Loads map data from disk.

    Expects the following directory structure:
        map_path/
        ├── segments_anno.json
        ├── clip_features.npy
        ├── point_cloud.pcd
        └── segments/
            └── {id}/
                └── rgb/
                    └── *.png
    """

    def __init__(self, map_path: str):
        """
        Initialize disk-based data source.
        """
        self.map_path = Path(map_path)
        self._annotations: Optional[List[Dict]] = None
        self._features: Optional[torch.Tensor] = None
        self._pcd: Optional[List[o3d.geometry.PointCloud]] = None
        self._bbox: Optional[List[o3d.geometry.OrientedBoundingBox]] = None
        self._classes: Optional[List[str]] = None

        log.info(f"Initialized DiskMapDataSource from {self.map_path}")

    def _load_annotations(self) -> List[Dict]:
        """Load annotations from segments_anno.json."""
        anno_path = self.map_path / "segments_anno.json"
        with open(anno_path, "r") as f:
            data = json.load(f)
        return data.get("segGroups", [])

    def _load_features(self) -> torch.Tensor:
        """Load CLIP features from clip_features.npy."""
        feat_path = self.map_path / "clip_features.npy"
        feats_np = np.load(feat_path)
        return torch.from_numpy(feats_np).float()

    def _load_point_clouds(self) -> List[o3d.geometry.PointCloud]:
        """Load per-object point clouds."""
        return load_point_cloud(self.map_path)

    def _load_classes(self) -> Optional[List[str]]:
        """Load class names from classes.json if available."""
        classes_path = self.map_path / "classes.json"
        if classes_path.exists():
            with open(classes_path, "r") as f:
                return json.load(f)
        return None

    @property
    def classes(self) -> Optional[List[str]]:
        if self._classes is None:
            self._classes = self._load_classes()
        return self._classes

    @property
    def annotations(self) -> List[Dict]:
        if self._annotations is None:
            self._annotations = self._load_annotations()
        return self._annotations

    @property
    def features(self) -> torch.Tensor:
        if self._features is None:
            self._features = self._load_features()
        return self._features

    @property
    def bbox(self) -> List[o3d.geometry.OrientedBoundingBox]:
        if self._bbox is None:
            if self._pcd is None:
                self._pcd = self._load_point_clouds()
            self._bbox = [pcd.get_oriented_bounding_box() for pcd in self._pcd]
        return self._bbox

    def get_object_images(self, object_id: int, limit: int = 5) -> List[np.ndarray]:
        """Load RGB images for a specific object from disk.

        Structure: map_dir/segments/{id}/rgb/*.png
        """
        rgb_path = self.map_path / "segments" / str(object_id) / "rgb" / "*.png"
        image_files = natsorted(glob.glob(str(rgb_path)))

        images = []
        for img_p in image_files[:limit]:
            img = cv2.imread(str(img_p))
            images.append(img)

        return images

    def get_object_centroid(self, object_id: int) -> np.ndarray:
        """Get centroid from annotations."""
        return np.array(self.annotations[object_id]["centroid"])

    def get_object_labels(self, object_id: int) -> List[str]:
        """Get the list of segment labels for a specific object.

        If classes.json is available, converts label indices to text labels.
        Otherwise returns the raw label values from annotations.
        """
        label_indices = self.annotations[object_id].get("labels", [])
        if not label_indices:
            return []

        # If classes list is available, convert indices to text labels
        if self.classes is not None:
            return [self.classes[int(idx)] for idx in label_indices]

        log.warning("Classes list not found, returning raw label indices as strings.")
        return [str(l) for l in label_indices]


class InMemoryMapDataSource(MapDataSource):
    """
    Uses an ObjectMap in memory without disk I/O.

    This is useful for real-time processing where the ObjectMap is already
    constructed (e.g., from perception pipeline in ProcTHORAgent).
    """

    def __init__(self, object_map: ObjectMap):
        """
        Initialize in-memory data source.
        """
        self.object_map = object_map
        self._annotations: Optional[List[Dict]] = None
        self._bbox: Optional[List[o3d.geometry.OrientedBoundingBox]] = None

        # Ensure map is collated
        if object_map.semantic_tensor is None:
            object_map.collate()

        log.info(f"Initialized InMemoryMapDataSource with {len(object_map)} objects")

    def _build_annotations(self) -> List[Dict]:
        """Build annotation dicts from ObjectMap objects."""
        annotations = []
        for i, obj in enumerate(self.object_map):
            ann = {
                "id": i,
                "objectId": i,
                "label": obj.tag if hasattr(obj, "tag") else "",
                "centroid": obj.centroid.tolist() if hasattr(obj.centroid, "tolist") else list(obj.centroid),
                "timestamps": [float(t) for t in obj.timestamps] if hasattr(obj, "timestamps") else [],
            }
            annotations.append(ann)
        return annotations

    @property
    def annotations(self) -> List[Dict]:
        if self._annotations is None:
            self._annotations = self._build_annotations()
        return self._annotations

    @property
    def features(self) -> torch.Tensor:
        return self.object_map.semantic_tensor

    @property
    def bbox(self) -> List[o3d.geometry.OrientedBoundingBox]:
        if self._bbox is None:
            self._bbox = self.object_map.oriented_bbox_o3d
        return self._bbox

    def get_object_images(self, object_id: int, limit: int = 5) -> List[np.ndarray]:
        """Get RGB images directly from ObjectMap segments."""
        obj = self.object_map[object_id]
        images = []

        # Get sorted segments and extract RGB images
        sorted_segments = obj.segments.get_sorted()
        for seg in sorted_segments[:limit]:
            if hasattr(seg, "rgb") and seg.rgb is not None:
                images.append(seg.rgb)

        return images

    def get_object_centroid(self, object_id: int) -> np.ndarray:
        """Get centroid directly from ObjectMap object."""
        return self.object_map[object_id].centroid

    def get_object_labels(self, object_id: int) -> List[str]:
        """Get labels directly from ObjectMap object segments.

        If classes list is available on the ObjectMap, converts label indices to text labels.
        Otherwise returns the raw label values as strings.
        """
        obj = self.object_map[object_id]
        label_indices = obj.labels

        # If classes list is available, convert indices to text labels
        if self.object_map.classes is not None:
            return [self.object_map.classes[int(idx)] for idx in label_indices]

        # Fall back to returning indices as strings
        log.warning("Classes list not found, returning raw label indices as strings.")
        return [str(l) for l in label_indices]