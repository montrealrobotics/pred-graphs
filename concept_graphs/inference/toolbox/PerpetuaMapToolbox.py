from typing import Dict
import logging
from pathlib import Path
import jax
import numpy as np

from perpetua2.data.procthor import read_gt_scene, fetch_gt_assignment

from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox
from concept_graphs.mapping.CanonicalVectors import Vector, CanonicalVectors
from concept_graphs.perception.ft_extraction.CLIP import CLIP
from concept_graphs.utils import ros_point_to_procthor

log = logging.getLogger(__name__)


class PerpetuaMapToolbox(ObjectMapToolbox):
    def __init__(
        self,
        map_path: str,
        ft_extractor: CLIP,
        data_path: str,
        p_m: float,
        p_f: float,
        alpha: float,
        decay_rate: float,
        map_time: int,
        load_week: bool = False,
        seed: int = 42,
    ):
        self.object_map: PerpetuaObjectMap = None
        self.key = jax.random.PRNGKey(seed)
        self.data_path = data_path
        self.gt_data = read_gt_scene(Path(data_path).parent / "groundtruth")
        self.p_m = p_m
        self.p_f = p_f
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.load_week = load_week
        self.map_time = map_time
        super().__init__(map_path, ft_extractor)

    @property
    def _estimator_kwargs(self):
        # We always use the same jax key for reproducibility!
        return {
            "data_path": self.data_path,
            "load_week": self.load_week,
            "p_m": self.p_m,
            "p_f": self.p_f,
            "alpha": self.alpha,
            "decay_rate": self.decay_rate,
            "key": self.key,
            "map_time": self.map_time,
        }

    def load_object_map(self, map_path: str = None):
        object_map = PerpetuaObjectMap.load(map_path)
        object_map.update_estimators(**self._estimator_kwargs)
        object_map.to(self.device)
        object_map.downsample_objects(voxel_size=0.01)
        return object_map

    def update_object_map(self, object_map):
        self.object_map = object_map
        object_map.update_estimators(**self._estimator_kwargs)
        self.object_map.downsample_objects(voxel_size=0.01)
        self.object_map.refresh_state()
        self.reset()

    def update_canonical_vectors(self, object_centroids_obs: Dict[str, Dict[str, np.ndarray]]):
        # Format observations
        vectors: Dict[str, CanonicalVectors] = {}
        for p_name, data in object_centroids_obs.items():
            for r_name, pickupable_centroid in data.items():
                receptacle_centroid = self.object_map.get_receptacle(r_name).centroid
                if r_name not in vectors:
                    vectors[r_name] = CanonicalVectors()
                vectors[r_name] += Vector.create(vector=pickupable_centroid - receptacle_centroid, object_id=p_name)
        # If nothing was updated, return
        if len(vectors) == 0:
            return
        self.object_map.update_canonical_vectors(vectors)

    def temporal_map_update(self, object_containment_obs: dict, obs_time: float):
        self.object_map.update(object_containment_obs, obs_time, self.p_m, self.p_f)

    def temporal_map_query(self, query_time: float):
        self.object_map.predict(query_time)
        self.object_map.refresh_state()

    def temporal_object_query(
        self,
        object_id: str,
        query_time: float,
        threshold: float = 0.5,
        add_noise: bool = True,
        condition_on_object: bool = False,
        strategy: str = "last_seen",
    ):
        prediction = self.object_map.object_predict(
            object_id,
            query_time,
            threshold,
            add_noise=add_noise,
            condition_on_object=condition_on_object,
            strategy=strategy,
        )
        gt_state = fetch_gt_assignment(self.gt_data, query_time, object_id, prediction)
        self.object_map.refresh_state()
        return prediction, gt_state

    def get_object_size_position(self, object_id: str, procthor: bool = True):
        obj_instance = self.object_map.get_object(object_id)
        centroid = obj_instance.centroid
        verices = obj_instance.vertices
        if procthor:
            centroid = ros_point_to_procthor(centroid)
            verices = ros_point_to_procthor(verices)
        extents = np.ptp(verices, axis=0)
        return extents, centroid.squeeze()

    def reset_temporal_edges(self):
        self.object_map.reset()

    def refresh_object_map(self, condition_on_object: bool, add_noise: bool, strategy: str = "last_seen"):
        self.object_map.set_edges(
            edges=None,
            move_pickupables=True,
            condition_on_object=condition_on_object,
            add_noise=add_noise,
            strategy=strategy,
        )

    # This is a debug method to visualize the canonical vectors
    def build_canonical_vectors(self):
        vectors_dict = {}
        for r_name, receptacle in self.object_map._receptacles.items():
            # All receptacles are assumed to have canonical vectors
            vectors_dict[r_name] = receptacle.canonical_pos_vectors
        return vectors_dict
