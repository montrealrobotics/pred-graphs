from typing import Dict, Any, Union, List, Optional

import jax.numpy as jnp

import logging

from concept_graphs.inference.agent.PrivilegedProcTHORAgent import PrivilegedProcTHORAgent
from concept_graphs.utils import procthor_to_ros, intrinsics_dict_to_matrix
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.inference.MapDataSource import InMemoryMapDataSource

log = logging.getLogger(__name__)


class ProcTHORAgent(PrivilegedProcTHORAgent):
    def __init__(
        self,
        perception_pipeline=None,
        mapping_factory=None,
        verifier=None,
        semantic_sim=None,
        query_engine=None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Perception components
        self.perception_pipeline = perception_pipeline
        self.mapping_factory = mapping_factory
        self.verifier = verifier
        self.semantic_sim = semantic_sim
        self.query_engine_partial = query_engine  # Partial config from Hydra
        self.device = device

        # Share ft_extractor from toolbox with perception_pipeline to avoid loading CLIP twice
        if self.perception_pipeline is not None and hasattr(self.toolbox, 'ft_extractor'):
            self.perception_pipeline.ft_extractor = self.toolbox.ft_extractor
            log.info("Sharing ft_extractor from toolbox with perception_pipeline")

        # Frame processing state
        self._last_frame_results: Optional[Dict] = None
        self._frame_processed: bool = False
        log.info("Agent is ready to be used!!!")

    def _reset_frame_cache(self):
        """Reset cached frame processing results."""
        self._last_frame_results = None
        self._frame_processed = False

    def _get_receptacles_bbox_from_map(self) -> Dict[str, Dict]:
        """Extract receptacle bboxes from perpetua map for MapQueryEngine.

        Returns: 
            Dict mapping receptacle names to their cornerPoints
        """
        result = {}
        for r_name in self.toolbox.object_map.get_receptacles_name():
            receptacle = self.toolbox.object_map.get_receptacle(r_name)
            result[r_name] = {"cornerPoints": receptacle.vertices.tolist()}
        return result

    def _get_p2r_map_from_pickupables(self) -> Dict[str, List[str]]:
        """Extract pickupable-to-receptacles mapping from perpetua map.

        Returns:
            Dict mapping pickupable names to list of possible receptacle names
        """
        result = {}
        for p_name in self.toolbox.object_map.get_pickupables_name():
            pickupable = self.toolbox.object_map.get_pickupable(p_name)
            result[p_name] = pickupable.receptacles
        return result

    # Frame processing methods
    def _process_frame(self, observation: Dict[str, Any]) -> Optional[Dict]:
        """Process a single frame through perception and query engine.

        Args:
            observation: Dictionary containing rgb_first_view, depth_view, agent_pose, intrinsics

        Returns:
            Dict with processing results or None if no objects detected
        """
        if self.perception_pipeline is None or self.mapping_factory is None:
            log.warning("Perception pipeline or mapping factory not configured")
            return None

        # 1. Extract and transform observation data
        rgb = observation["rgb_first_view"]
        depth = observation["depth_view"].squeeze()
        agent_pose = observation["agent_pose"]
        intrinsics_dict = observation["intrinsics"]

        # 2. Convert intrinsics and pose
        intrinsics_matrix = intrinsics_dict_to_matrix(intrinsics_dict)
        camera_pose = procthor_to_ros(agent_pose)

        # 3. Run perception pipeline
        segments = self.perception_pipeline(rgb, depth, intrinsics_matrix, camera_pose)
        if segments is None:
            log.info("No segments detected in frame")
            return None

        # 4. Create local map
        local_map = self.mapping_factory()
        local_map.from_perception(**segments, timestamp=self.current_time)
        local_map.check_processing()
        local_map.classes = list(self.perception_pipeline.segmentation_model.classes)


        if len(local_map) == 0:
            log.info("No objects in local map after perception")
            return None

        # 5. Run query engine
        return self._run_query_engine(local_map)

    def _run_query_engine(self, local_map: ObjectMap) -> Dict:
        """
        Run MapQueryEngine on local map using in-memory data source.
        """
        data_source = InMemoryMapDataSource(local_map)

        # Initialize query engine with runtime parameters
        # The query_engine_partial is a partial function from Hydra config
        engine = self.query_engine_partial(
            data_source=data_source,
            ft_extractor=self.toolbox.ft_extractor,
            semantic_sim=self.semantic_sim,
            verifier=self.verifier,
            receptacles_bbox=self._get_receptacles_bbox_from_map(),
            pickupable_to_receptacles=self._get_p2r_map_from_pickupables(),
        )

        # Run queries for all pickupable names
        pickupable_names = self.toolbox.object_map.get_pickupables_name()
        results = engine.process_queries(queries=pickupable_names)

        pickupable_map_ids, receptacle_map_ids = engine.get_map_object_ids()
        # from receptacle_map_ids remove the keys where value is None
        receptacle_map_ids = {k: v for k, v in receptacle_map_ids.items() if v is not None}
        assignments = engine.parse_assignments(results, pickupable_names)

        return {
            "results": results,
            "pickupable_map_ids": pickupable_map_ids,
            "receptacle_map_ids": receptacle_map_ids,
            "assignments": assignments,
            "local_map": local_map,
            "detected_pickupables": set(k for k, v in assignments.items() if v is not None),
            "detected_receptacles": set(receptacle_map_ids.keys()),
        }

    # Main agent methods
    def update(self, observation: Dict[str, Any]) -> None:
        """
        Updates the internal belief state based on a new environment observation.
        Uses perception pipeline and MapQueryEngine for non-privileged detection.

        Args:
            observation (Dict[str, Any]): Dictionary containing:
                - rgb_first_view: RGB image
                - depth_view: Depth map
                - agent_pose: Agent pose dict
                - intrinsics: Camera intrinsics dict

        Returns:
            None
        """
        if not self._frame_processed or self._last_frame_results is None:
            log.info("Processing new frame for update")
            frame_results = self._process_frame(observation)
            if frame_results is None:
                log.info("No frame results - skipping found_pickupable")
                return
        else:
            frame_results = self._last_frame_results

        if frame_results is None:
            log.info("No frame results - skipping update")
            return

        # Cache results for found_pickupable
        self._last_frame_results = frame_results
        self._frame_processed = True

        # Extract what we need
        assignments = frame_results["assignments"]
        pickupable_map_ids = frame_results["pickupable_map_ids"]
        receptacle_map_ids = frame_results["receptacle_map_ids"]
        local_map = frame_results["local_map"]
        detected_pickupables = frame_results["detected_pickupables"]
        visible_receptacles = frame_results["detected_receptacles"]

        # Get p2r mapping for all pickupables
        p2r_map = self._get_p2r_map_from_pickupables()

        # Build observation dictionaries
        object_containment_obs = {}
        object_centroids_obs = {}

        # For detected pickupables
        for p_name, receptacle_name in assignments.items():
            if receptacle_name is None:
                continue

            # Skip pickupables that are not in the map
            if not self.toolbox.object_map.has_object(p_name):
                continue
            pickupable = self.toolbox.object_map.get_pickupable(p_name)
            potential_receptacles = pickupable.receptacles

            obs_receptacle = {}
            obs_centroid = {}

            for r_name in potential_receptacles:
                if r_name == "OOB_FAKE_RECEPTACLE":
                    continue
                if r_name == receptacle_name:
                    obs_receptacle[r_name] = jnp.array([1.0])  # Positive observation
                    # Get centroid from local map
                    obj_idx = pickupable_map_ids[p_name]
                    obs_centroid[r_name] = local_map[obj_idx].centroid
                else: # All other potential receptacles get negative observation
                    obs_receptacle[r_name] = jnp.array([0.0])  # Negative observation

            if obs_receptacle:
                object_containment_obs[p_name] = obs_receptacle
            if obs_centroid:
                object_centroids_obs[p_name] = obs_centroid

        # For visible receptacles without their pickupables (negative observations)
        for r_name in visible_receptacles:
            if r_name == "OOB_FAKE_RECEPTACLE":
                continue
            # Find all pickupables that could be on this receptacle
            for p_name, potential_rs in p2r_map.items():
                if r_name in potential_rs and p_name not in detected_pickupables:
                    if p_name not in object_containment_obs:
                        object_containment_obs[p_name] = {}
                    object_containment_obs[p_name][r_name] = jnp.array([0.0])

        log.info(f"Object containment obs: {object_containment_obs}")

        # Update temporal map
        self.toolbox.temporal_map_update(object_containment_obs, self.current_time)
        self.toolbox.update_canonical_vectors(object_centroids_obs)

        # Free memory by deleting the local map
        if self._last_frame_results is not None and "local_map" in self._last_frame_results:
            del self._last_frame_results["local_map"]

    def found_pickupable(
        self, receptacle_id: str, pickupable_id: str, observation: Dict[str, Any],
    ) -> Union[bool, None]:
        """
        Checks if the current observation contains the pickupable of interest.
        Uses cached frame processing results from the last update() call.

        Args:
            receptacle_id (str): The id of the target receptacle.
            pickupable_id (str): The id of the object we are searching for.
            observation (Dict[str, Any]): The observation dictionary for the current state.

        Returns:
            bool | None:
                - True if the object is detected at the specified receptacle.
                - False if the receptacle is visible but the object is not present at that receptacle.
                - None if the receptacle is not visible in the current observation.
        """
        # Use cached frame processing results if they exist
        # Otherwise process new frame
        if not self._frame_processed or self._last_frame_results is None:
            log.info("Processing new frame for found")
            frame_results = self._process_frame(observation)
            if frame_results is None:
                log.info("No frame results - skipping found_pickupable")
                return
            self._last_frame_results = frame_results
            self._frame_processed = True

        detected_pickupables = self._last_frame_results.get("detected_pickupables", set())
        detected_receptacles = self._last_frame_results.get("detected_receptacles", set())

        if pickupable_id in detected_pickupables:
            return True

        receptacles_for_p = self.toolbox.object_map.get_pickupable(pickupable_id).receptacles
        # Pickupable not detected - check if receptacle is visible
        if any(r in receptacles_for_p for r in detected_receptacles):
            return False  # Receptacle visible but pickupable not there

        return None  # Receptacle not visible
