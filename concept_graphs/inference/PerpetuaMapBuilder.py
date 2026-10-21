import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import dill as pickle

from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from concept_graphs.mapping.CanonicalVectors import Vector, CanonicalVectors

log = logging.getLogger(__name__)


class PerpetuaMapBuilder:
    def __init__(
        self,
        estimators_path: str,
        output_path: str,
        full_overwrite: bool = False,
    ):
        """
        Initialize the PerpetuaMap builder.
        """
        self.estimators_path = Path(estimators_path)
        self.output_path = Path(output_path)
        self.full_overwrite = full_overwrite

        log.info(f"PerpetuaMapBuilder initialized with output_path={output_path}")

    def _load_estimator(self, pickupable_name: str):
        """
        Load the Perpetua estimator for a given pickupable object.
        """
        est_path = (
            self.estimators_path
            / f"{pickupable_name.replace('|', '_')}_object.pkl"
        )
        try:
            with open(est_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading estimator from {est_path}: {e}")

    def _build_canonical_vectors(
        self,
        concept_nodes_map: ObjectMap,
        current_p2r_mapping: Dict[str, Optional[str]],
        pickupable_map_ids: Dict[str, Optional[int]],
        receptacle_map_ids: Dict[str, Optional[int]],
    ) -> Dict[str, CanonicalVectors]:
        """
        Build canonical vectors from pickupable-receptacle relationships.

        Canonical vectors represent the spatial relationship between a pickupable
        and its containing receptacle.
        """
        canonical_vectors: Dict[str, CanonicalVectors] = {}

        for pickupable_name, receptacle_name in current_p2r_mapping.items():
            if receptacle_name is None:
                continue

            pickupable_id = pickupable_map_ids.get(pickupable_name)
            receptacle_id = receptacle_map_ids.get(receptacle_name)

            if pickupable_id is None or receptacle_id is None:
                continue

            pickupable_obj = concept_nodes_map[pickupable_id]
            receptacle_obj = concept_nodes_map[receptacle_id]

            np_vector = np.array(
                pickupable_obj.centroid - receptacle_obj.centroid,
                dtype=np.float32
            )
            vector = Vector.create(np_vector, pickupable_name)

            if receptacle_name not in canonical_vectors:
                canonical_vectors[receptacle_name] = CanonicalVectors()
            canonical_vectors[receptacle_name] += vector

        return canonical_vectors

    def _build_edges(
        self,
        current_p2r_mapping: Dict[str, Optional[str]],
        perpetua_map: PerpetuaObjectMap,
        skip_pickupables: Optional[Set[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Build edge graph from pickupable-receptacle relationships.
        """
        skip = skip_pickupables or set()
        edges: Dict[str, List[str]] = {}

        for pickupable_name, receptacle_name in current_p2r_mapping.items():
            if receptacle_name is None or pickupable_name in skip:
                continue
            if not perpetua_map.has_object(pickupable_name):
                continue
            if not perpetua_map.has_object(receptacle_name):
                continue

            if receptacle_name not in edges:
                edges[receptacle_name] = []
            edges[receptacle_name].append(pickupable_name)

        return edges

    def build_or_update(
        self,
        time: float,
        concept_nodes_map: ObjectMap,
        pickupable_names: List[str],
        receptacle_names: List[str],
        pickupable_map_ids: Dict[str, Optional[int]],
        receptacle_map_ids: Dict[str, Optional[int]],
        current_p2r_mapping: Dict[str, Optional[str]],
    ) -> PerpetuaObjectMap:
        """
        Build a new PerpetuaObjectMap or update an existing one.
        """
        # Remove OOB_FAKE_RECEPTACLE from inputs
        receptacle_map_ids = dict(receptacle_map_ids)  # Copy to avoid modifying original
        if "OOB_FAKE_RECEPTACLE" in receptacle_map_ids:
            del receptacle_map_ids["OOB_FAKE_RECEPTACLE"]

        receptacle_names = list(receptacle_names)  # Copy to avoid modifying original
        if "OOB_FAKE_RECEPTACLE" in receptacle_names:
            receptacle_names.remove("OOB_FAKE_RECEPTACLE")

        map_file = self.output_path / "perpetua_map.pkl"

        # Build canonical vectors
        canonical_vectors = self._build_canonical_vectors(
            concept_nodes_map,
            current_p2r_mapping,
            pickupable_map_ids,
            receptacle_map_ids,
        )

        if map_file.exists() and not self.full_overwrite:
            # Update existing map
            perpetua_map = PerpetuaObjectMap.load(self.output_path)
            perpetua_map = self._update_existing_map(
                perpetua_map,
                concept_nodes_map,
                pickupable_names,
                receptacle_names,
                pickupable_map_ids,
                receptacle_map_ids,
                canonical_vectors,
            )
        else:
            # Create new map
            perpetua_map = self._create_new_map(
                time,
                concept_nodes_map,
                pickupable_names,
                receptacle_names,
                pickupable_map_ids,
                receptacle_map_ids,
                current_p2r_mapping,
                canonical_vectors,
            )

        perpetua_map.save(self.output_path)
        return perpetua_map

    def _update_existing_map(
        self,
        perpetua_map: PerpetuaObjectMap,
        concept_nodes_map: ObjectMap,
        pickupable_names: List[str],
        receptacle_names: List[str],
        pickupable_map_ids: Dict[str, Optional[int]],
        receptacle_map_ids: Dict[str, Optional[int]],
        canonical_vectors: Dict[str, CanonicalVectors],
    ) -> PerpetuaObjectMap:
        """
        Update an existing PerpetuaObjectMap with new objects.
        """
        # Add new pickupables
        for pickupable_name in pickupable_names:
            pickupable_id = pickupable_map_ids.get(pickupable_name)
            if pickupable_id is None or perpetua_map.has_object(pickupable_name):
                continue

            estimator = self._load_estimator(pickupable_name)
            pickupable_obj = concept_nodes_map[pickupable_id]
            perpetua_map.add_pickupable(
                pickupable_name,
                pickupable_obj,
                estimator,
                visibility=False,
            )

        # Add new receptacles
        for receptacle_name in receptacle_names:
            receptacle_id = receptacle_map_ids.get(receptacle_name)
            if receptacle_id is None or perpetua_map.has_object(receptacle_name):
                continue

            receptacle_obj = concept_nodes_map[receptacle_id]
            perpetua_map.add_receptacle(receptacle_name, receptacle_obj)

        perpetua_map.update_canonical_vectors(canonical_vectors)
        perpetua_map.refresh_state()

        return perpetua_map

    def _create_new_map(
        self,
        time: float,
        concept_nodes_map: ObjectMap,
        pickupable_names: List[str],
        receptacle_names: List[str],
        pickupable_map_ids: Dict[str, Optional[int]],
        receptacle_map_ids: Dict[str, Optional[int]],
        current_p2r_mapping: Dict[str, Optional[str]],
        canonical_vectors: Dict[str, CanonicalVectors],
    ) -> PerpetuaObjectMap:
        """
        Create a new PerpetuaObjectMap from scratch.
        """
        perpetua_map = PerpetuaObjectMap(pickupable_names, receptacle_names, time)

        # Add receptacles first
        for receptacle_name in receptacle_names:
            receptacle_id = receptacle_map_ids.get(receptacle_name)
            if receptacle_id is None:
                continue

            receptacle_obj = concept_nodes_map[receptacle_id]
            perpetua_map.add_receptacle(receptacle_name, receptacle_obj)

        # Add pickupables
        for pickupable_name in pickupable_names:
            pickupable_id = pickupable_map_ids.get(pickupable_name)
            if pickupable_id is None:
                continue

            estimator = self._load_estimator(pickupable_name)
            pickupable_obj = concept_nodes_map[pickupable_id]
            perpetua_map.add_pickupable(
                pickupable_name,
                pickupable_obj,
                estimator,
                visibility=True,
            )

        # Add background objects (not pickupable or receptacle)
        used_object_keys = set()
        for idx in pickupable_map_ids.values():
            if idx is not None:
                used_object_keys.add(concept_nodes_map.key_map[idx])
        for idx in receptacle_map_ids.values():
            if idx is not None:
                used_object_keys.add(concept_nodes_map.key_map[idx])

        for obj_key, obj in concept_nodes_map.objects.items():
            if obj_key not in used_object_keys:
                background_name = f"background_{obj_key}"
                perpetua_map.add_background(background_name, obj)

        # Set up relationships
        perpetua_map.update_canonical_vectors(canonical_vectors)
        initial_edges = self._build_edges(current_p2r_mapping, perpetua_map)
        perpetua_map.set_edges(initial_edges, move_pickupables=False)
        perpetua_map.set_initial_edges()
        perpetua_map.refresh_state()

        return perpetua_map
