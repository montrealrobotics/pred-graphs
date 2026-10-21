import logging
import shutil
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

import torch
from tqdm import tqdm
import open3d as o3d

from concept_graphs.vlm.OpenAIVerifier import OpenAIVerifier
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from concept_graphs.perception.ft_extraction.FeatureExtractor import FeatureExtractor
from concept_graphs.utils import split_camel_preserve_acronyms, aabb_iou

from .MapDataSource import MapDataSource, DiskMapDataSource

from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


class MapQueryEngine:
    def __init__(
        self,
        ft_extractor: FeatureExtractor,
        semantic_sim: CosineSimilarity01,
        verifier: OpenAIVerifier,
        receptacles_bbox: Dict,
        pickupable_to_receptacles: Dict,
        map_path: Optional[str] = None,
        data_source: Optional[MapDataSource] = None,
        pickupable_bbox: Optional[Dict] = None,
        pickupable_existence: Optional[Dict] = None,
        device: str = "cuda",
        top_k: int = 3,
        use_hungarian: bool = False,
        use_vlm_verification: bool = True,
        receptacle_breakaway_margin: float = 0.05,
        min_dist_sq: float = 1.0,
    ):
        """
        Initialize the query engine.

        Note: Provide either map_path OR data_source, not both.
        """
        # Create data source from map_path if not provided directly
        if data_source is not None:
            self.data_source = data_source
        elif map_path is not None:
            self.data_source = DiskMapDataSource(map_path)
        else:
            raise ValueError("Either map_path or data_source must be provided")

        self.ft_extractor = ft_extractor
        self.semantic_sim = semantic_sim
        self.verifier = verifier
        self.receptacles_bbox = receptacles_bbox
        self.pickupable_to_receptacles = pickupable_to_receptacles
        self.device = device
        self.top_k = top_k
        self.use_hungarian = use_hungarian
        self.use_vlm_verification = use_vlm_verification
        self.receptacle_breakaway_margin = receptacle_breakaway_margin
        self.min_dist_sq = min_dist_sq
        # For GT subclass
        self.pickupable_bbox = pickupable_bbox or {}
        self.pickupable_existence = pickupable_existence or {}

        # Containers for map IDs
        self.pickupable_map_ids: Dict[str, int] = {}
        self.gt_receptacles_aabb = self._compute_receptacles_aabb()
        self.receptacle_map_ids = self._compute_receptacle_map_ids()

        log.info(f"MapQueryEngine initialized with {len(self.data_source.annotations)} objects")

    def _compute_receptacles_aabb(self) -> Dict[str, o3d.geometry.AxisAlignedBoundingBox]:
        """Compute axis-aligned bounding boxes for all receptacles."""
        result = {}
        for r_name, r_data in self.receptacles_bbox.items():
            corners = np.array(r_data["cornerPoints"], dtype=np.float32)
            oobb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners)
            )
            aabb = oobb.get_axis_aligned_bounding_box()
            result[r_name] = aabb
        return result

    def _canonical_receptacle_key(self, key: str) -> str:
        """Extract canonical key (physical object id) from receptacle key."""
        return key.split("___", 1)[0]

    def _compute_receptacle_map_ids(self) -> Dict[str, Optional[int]]:
        """Assign each receptacle to a map bbox index using Hungarian matching."""
        receptacle_to_map: Dict[str, Optional[int]] = {}
        bbox = self.data_source.bbox

        # Collect receptacles (skip fake)
        receptacle_keys = [
            k for k in self.receptacles_bbox.keys() if k != "OOB_FAKE_RECEPTACLE"
        ]
        receptacle_to_map["OOB_FAKE_RECEPTACLE"] = None

        if not receptacle_keys or not bbox:
            return receptacle_to_map

        # Use precomputed AABBs to get receptacle corners
        rec_corners_list: List[np.ndarray] = [
            np.asarray(self.gt_receptacles_aabb[k].get_box_points(), dtype=np.float32)
            for k in receptacle_keys
        ]

        # Precompute bbox corners (convert oriented to axis-aligned for consistent AABB IoU)
        bbox_corners_list: List[np.ndarray] = [
            np.asarray(b.get_axis_aligned_bounding_box().get_box_points(), dtype=np.float32)
            for b in bbox
        ]

        R = len(receptacle_keys)
        B = len(bbox_corners_list)

        # IoU matrix: (R x B)
        iou_rb = np.zeros((R, B), dtype=np.float32)
        for i in range(R):
            rc = rec_corners_list[i]
            for j in range(B):
                iou_rb[i, j] = aabb_iou(rc, bbox_corners_list[j])

        # Group receptacles by canonical key
        canon_to_indices: Dict[str, List[int]] = {}
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)
            canon_to_indices.setdefault(ck, []).append(i)

        canon_keys = list(canon_to_indices.keys())
        G = len(canon_keys)

        # Score matrix for groups vs bboxes: take best IoU among members of the group
        score_gb = np.zeros((G, B), dtype=np.float32)
        for gi, ck in enumerate(canon_keys):
            idxs = canon_to_indices[ck]
            score_gb[gi, :] = np.max(iou_rb[idxs, :], axis=0)

        # Allow "unassigned" by adding dummy columns
        n_cols = max(B, G)
        cost = np.ones((G, n_cols), dtype=np.float32)
        cost[:, :B] = 1.0 - score_gb

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # Determine each canonical group's primary bbox
        bbox_owner: Dict[int, str] = {}
        canon_primary_bbox: Dict[str, Optional[int]] = {}

        for gi, cj in zip(row_ind, col_ind):
            ck = canon_keys[gi]
            cj = int(cj)  # Convert numpy int64 to Python int
            if cj >= B:
                canon_primary_bbox[ck] = None
                continue
            if score_gb[gi, cj] < 1e-3:
                canon_primary_bbox[ck] = None
                continue
            canon_primary_bbox[ck] = cj
            bbox_owner[cj] = ck

        # Initial per-receptacle mapping
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)
            receptacle_to_map[k] = canon_primary_bbox.get(ck, None)

        # Refinement: allow a receptacle to switch to a different bbox
        for i, k in enumerate(receptacle_keys):
            ck = self._canonical_receptacle_key(k)

            current = receptacle_to_map[k]
            current_iou = iou_rb[i, current] if (current is not None) else 0.0

            best_j = int(np.argmax(iou_rb[i, :]))
            best_iou = float(iou_rb[i, best_j])

            if best_iou < 1e-3:
                continue

            owner = bbox_owner.get(best_j, None)
            if owner is not None and owner != ck:
                continue

            if (best_iou - current_iou) >= self.receptacle_breakaway_margin:
                receptacle_to_map[k] = best_j
                bbox_owner[best_j] = ck

        return receptacle_to_map

    def _point_to_aabb_distance(self, point: np.ndarray, corners: np.ndarray) -> float:
        """Compute distance from a point to an axis-aligned bounding box."""
        b_min = np.min(corners, axis=0)
        b_max = np.max(corners, axis=0)
        d_vars = np.maximum(0, b_min - point) + np.maximum(0, point - b_max)
        dist_sq = np.sum(d_vars**2)
        return np.sqrt(dist_sq)

    def _find_receptacle(
        self, object_centroid: List[float], receptacles: Dict, only_in_map: bool = False
    ) -> Optional[str]:
        """
        Find the closest receptacle to an object centroid. If only_in_map is True, only
        consider receptacles that exist in the map (i.e., have a map_object_id).
        """
        if not receptacles:
            return None

        map_point = np.array(object_centroid)
        closest_rec = None
        min_dist_sq = self.min_dist_sq

        for rec_name, rec_data in receptacles.items():
            # Skip receptacles not detected in map if only_in_map is True
            if only_in_map and self.receptacle_map_ids.get(rec_name) is None:
                continue

            corners = np.asarray(rec_data.get_box_points(), dtype=np.float32)
            dist_sq = self._point_to_aabb_distance(map_point, corners)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_rec = rec_name

        return closest_rec

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries using CLIP text encoder."""
        log.info("Encoding queries...")
        queries_cleaned = [
            split_camel_preserve_acronyms(q.split("|")[0]) for q in queries
        ]
        return self.ft_extractor.encode_text(queries_cleaned).cpu()

    def _build_receptacles_list(self, query_text: str) -> List[Dict]:
        """Build the receptacles list for a given query."""
        receptacles = []
        receptacle_names = self.pickupable_to_receptacles.get(query_text, [])

        for rec_name in receptacle_names:
            if rec_name == "OOB_FAKE_RECEPTACLE":
                continue

            rec_map_id = self.receptacle_map_ids.get(rec_name)
            if rec_map_id is None:
                receptacles.append({
                    "receptacle_name": rec_name,
                    "map_object_id": None,
                    "receptacle_timestamps": [],
                })
                continue

            rec_timestamps = self.data_source.annotations[rec_map_id].get("timestamps", [])
            receptacles.append({
                "receptacle_name": rec_name,
                "map_object_id": rec_map_id,
                "receptacle_timestamps": rec_timestamps,
            })

        return receptacles

    def _create_empty_result_entry(self, query_text: str) -> Dict:
        """Create an empty result entry with receptacles populated."""
        return {
            "present": False,
            "map_object_id": None,
            "query_timestamp": [],
            "present_receptacle_name": None,
            "receptacles": self._build_receptacles_list(query_text),
        }

    def _get_allowed_object_indices(self) -> List[int]:
        """Get indices of map objects that are not receptacles."""
        forbidden = {v for v in self.receptacle_map_ids.values() if v is not None}
        return [j for j in range(len(self.data_source.bbox)) if j not in forbidden]

    def _verify_candidate(
        self, idx: int, query_text: str, check_receptacle: bool = True
    ) -> bool:
        """Verify if a candidate object matches the query using the VLM verifier."""
        images = self.data_source.get_object_images(idx, limit=self.verifier.max_images)
        if len(images) == 0:
            log.warning(f"No images found for object ID {idx}. Skipping.")
            return False

        query_text_cleaned = split_camel_preserve_acronyms(query_text.split("|")[0])
        is_match = self.verifier(images, query_text_cleaned)

        if is_match and check_receptacle and idx in self.receptacle_map_ids.values():
            log.warning(
                f"Verified object ID {idx} for query '{query_text}' is actually a receptacle. Skipping."
            )
            return False

        return is_match

    def _fill_result_entry(self, result_entry: Dict, idx: int) -> None:
        """Fill in the result entry with object data for a verified match."""
        obj_data = self.data_source.annotations[idx]
        centroid = obj_data["centroid"]
        rec_name = self._find_receptacle(centroid, self.gt_receptacles_aabb)

        # Warn if the matched receptacle doesn't exist in the map
        if rec_name is not None and self.receptacle_map_ids.get(rec_name) is None:
            log.warning(
                f"Pickupable at map object {idx} matched to receptacle '{rec_name}' "
                f"which was not detected in the map (no map_object_id)."
            )

        result_entry["present"] = True
        result_entry["map_object_id"] = idx
        result_entry["query_timestamp"] = obj_data.get("timestamps", [])
        result_entry["present_receptacle_name"] = rec_name

    def _compute_similarity_matrix(
        self, text_features: torch.Tensor, object_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Compute similarity matrix between queries and map objects."""
        text_features = text_features.to(self.device)
        features = self.data_source.features

        if object_indices is not None:
            features = features[object_indices].to(self.device)
        else:
            features = features.to(self.device)

        sim_matrix = self.semantic_sim(text_features, features).cpu()
        return sim_matrix

    def process_queries(self, queries: List[str]) -> Dict:
        """Process queries to find pickupable objects in the map.

        Uses either sequential top-k search or Hungarian matching based on
        the use_hungarian flag set during initialization.

        Returns:
            Dict mapping query text to result entries with:
            - present: bool
            - map_object_id: int | None
            - query_timestamp: List[float]
            - present_receptacle_name: str | None
            - receptacles: List[Dict]
        """
        if self.use_hungarian:
            return self._process_queries_hungarian(queries)
        else:
            return self._process_queries_sequential(queries)

    def _process_queries_sequential(self, queries: List[str]) -> Dict:
        """Sequential top-k search: for each query, verify candidates in order."""
        results: Dict[str, Dict] = {}
        pickupable_to_map: Dict[str, int] = {}

        text_features = self._encode_queries(queries)

        for i, query_text in tqdm(
            enumerate(queries), total=len(queries), desc="Processing Queries (Sequential)"
        ):
            query_feat = text_features[i].unsqueeze(0)
            sim_scores = self._compute_similarity_matrix(query_feat).squeeze()

            top_k_indices = torch.argsort(sim_scores, descending=True)[: self.top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            result_entry = self._create_empty_result_entry(query_text)

            for idx in top_k_indices:
                idx = int(idx)
                if self.use_vlm_verification:
                    if self._verify_candidate(idx, query_text):
                        self._fill_result_entry(result_entry, idx)
                        pickupable_to_map[query_text] = idx
                        break
                else:
                    # Accept match without verification
                    self._fill_result_entry(result_entry, idx)
                    pickupable_to_map[query_text] = idx
                    break

            results[query_text] = result_entry

        self.pickupable_map_ids = pickupable_to_map
        return results

    def _process_queries_hungarian(self, queries: List[str]) -> Dict:
        """Hungarian matching over CLIP embeddings for one-to-one assignment."""
        results: Dict[str, Dict] = {}
        pickupable_to_map: Dict[str, int] = {}

        allowed_indices = self._get_allowed_object_indices()

        if len(allowed_indices) == 0:
            log.warning("No non-receptacle objects found in map. Returning empty results.")
            for query_text in queries:
                results[query_text] = self._create_empty_result_entry(query_text)
            self.pickupable_map_ids = pickupable_to_map
            return results

        text_features = self._encode_queries(queries)
        sim_matrix = self._compute_similarity_matrix(text_features, allowed_indices)

        Q = len(queries)
        A = len(allowed_indices)

        # Compute top-k masks
        top_k_masks = np.zeros((Q, A), dtype=bool)
        for i in range(Q):
            top_k_idx = torch.argsort(sim_matrix[i], descending=True)[: self.top_k]
            top_k_masks[i, top_k_idx.numpy()] = True

        # Build cost matrix
        sim_np = sim_matrix.numpy()
        cost = 1.0 - sim_np
        cost[~top_k_masks] = 1e6

        n_cols = max(A, Q)
        cost_padded = np.ones((Q, n_cols), dtype=np.float32)
        cost_padded[:, :A] = cost

        row_ind, col_ind = linear_sum_assignment(cost_padded)

        query_to_candidate: Dict[str, Optional[int]] = {}
        for r, c in zip(row_ind, col_ind):
            query_text = queries[r]
            if c >= A:
                query_to_candidate[query_text] = None
            elif not top_k_masks[r, c]:
                query_to_candidate[query_text] = None
            else:
                query_to_candidate[query_text] = allowed_indices[c]

        for query_text in tqdm(queries, desc="Processing Queries (Hungarian)"):
            result_entry = self._create_empty_result_entry(query_text)
            candidate_idx = query_to_candidate.get(query_text)

            if candidate_idx is not None:
                if self.use_vlm_verification:
                    if self._verify_candidate(candidate_idx, query_text, check_receptacle=False):
                        self._fill_result_entry(result_entry, candidate_idx)
                        pickupable_to_map[query_text] = candidate_idx
                else:
                    # Accept match without verification
                    self._fill_result_entry(result_entry, candidate_idx)
                    pickupable_to_map[query_text] = candidate_idx

            results[query_text] = result_entry

        self.pickupable_map_ids = pickupable_to_map
        return results

    def get_map_object_ids(self) -> Tuple[Dict[str, int], Dict[str, Optional[int]]]:
        """Get both pickupable and receptacle map IDs."""
        return self.pickupable_map_ids, self.receptacle_map_ids

    def parse_assignments(
        self, assignments: Dict, pickupable_names: List[str]
    ) -> Dict[str, Optional[str]]:
        """Parse assignments into pickupable->receptacle mapping."""
        return {
            name: (
                assignments[name].get("present_receptacle_name")
                if assignments.get(name, {}).get("present")
                else None
            )
            for name in pickupable_names
        }

    def save_results(self, results, output_path):
        """Save results."""
        parsed_assignments = {}
        for p_name, res_data in results.items():
            present = res_data["present"]
            present_r_name = res_data["present_receptacle_name"]
            parsed_assignments[p_name] = {}
            # Case one, the pickupable was found and we share timestamp information
            if present and present_r_name is not None and present_r_name != "OOB_FAKE_RECEPTACLE":
                # Collect all timestamps where pickupable and receptacle has been observed!
                timestamps = res_data["query_timestamp"]
                for r_data in res_data["receptacles"]:
                    timestamps += r_data["receptacle_timestamps"]
                timestamps = np.array(sorted(timestamps))
                for r_data in res_data["receptacles"]:
                    r_name = r_data["receptacle_name"]
                    # For the present receptacle, all timestamps are valid
                    states = [1] * len(timestamps) if r_name == present_r_name else [0] * len(timestamps)
                    parsed_assignments[p_name][r_name] = {
                        "timestamps": timestamps.tolist(),
                        "states": states,
                        "map_object_id": r_data["map_object_id"],   
                    }
            # Object was not found, so we use each receptacle's timestamps to mark absence
            else:
                for r_data in res_data["receptacles"]:
                    r_name = r_data["receptacle_name"]
                    if r_name == "OOB_FAKE_RECEPTACLE":
                        continue
                    if len(r_data["receptacle_timestamps"]) == 0:
                        continue
                    timestamps = sorted(r_data["receptacle_timestamps"])
                    states = [0] * len(timestamps)
                    parsed_assignments[p_name][r_name] = {
                        "timestamps": timestamps,
                        "states": states,
                        "map_object_id": r_data["map_object_id"],   
                    }

        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "receptacle_map_ids.json", "w") as f:
            json.dump(self.receptacle_map_ids, f, indent=4)

        pickupable_map_ids = {}
        for p_name, res_data in results.items():
            if res_data["present"]:
                pickupable_map_ids[p_name] = res_data["map_object_id"]
        with open(output_path / "pickupable_map_ids.json", "w") as f:
            json.dump(pickupable_map_ids, f, indent=4)

        with open(output_path / "query_results.json", "w") as f:
            json.dump(results, f, indent=4)

        with open(output_path / "parsed_assignments.json", "w") as f:
            json.dump(parsed_assignments, f, indent=4)

    def visualize(self, object_map_path: str, results: Dict):
        """Visualize all object point clouds with labels."""
        from concept_graphs.inference.toolbox.ObjectMapToolbox import ObjectMapToolbox
        from concept_graphs.viz.server.VerifierServer import VerifierServer

        toolbox = ObjectMapToolbox(object_map_path, self.ft_extractor)
        server = VerifierServer(
            self.pickupable_bbox,
            self.receptacles_bbox,
            self.receptacle_map_ids,
            self.pickupable_existence,
            results,
            toolbox,
        )
        server.spin()


class MapQueryEngineGT(MapQueryEngine):
    """Ground-truth based query engine that uses GT bounding boxes instead of CLIP."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pickupable_map_ids = self._compute_pickupable_map_ids()

    def _compute_pickupable_map_ids(self) -> Dict[str, Optional[int]]:
        """One-to-one assignment of pickupables -> map bbox using Hungarian matching."""
        pickupable_to_map: Dict[str, Optional[int]] = {}
        bbox = self.data_source.bbox

        p_keys_all = list(self.pickupable_bbox.keys())
        p_keys = [k for k in p_keys_all if self.pickupable_existence.get(k, False)]

        for k in p_keys_all:
            if not self.pickupable_existence.get(k, False):
                pickupable_to_map[k] = None

        if len(p_keys) == 0 or len(bbox) == 0:
            for k in p_keys:
                pickupable_to_map[k] = None
            return pickupable_to_map

        allowed_bbox_indices = self._get_allowed_object_indices()

        if len(allowed_bbox_indices) == 0:
            for k in p_keys:
                pickupable_to_map[k] = None
            return pickupable_to_map

        P = len(p_keys)
        A = len(allowed_bbox_indices)

        # Convert oriented bboxes to axis-aligned for consistent AABB IoU
        allowed_bbox_corners: List[np.ndarray] = []
        for j in allowed_bbox_indices:
            allowed_bbox_corners.append(
                np.asarray(bbox[j].get_axis_aligned_bounding_box().get_box_points(), dtype=np.float32)
            )

        iou_pa = np.zeros((P, A), dtype=np.float32)
        for i, p_key in enumerate(p_keys):
            p_corners = np.array(
                self.pickupable_bbox[p_key]["cornerPoints"], dtype=np.float32
            )
            for a, box_pts in enumerate(allowed_bbox_corners):
                iou_pa[i, a] = aabb_iou(p_corners, box_pts)

        n_cols = max(A, P)
        cost = np.ones((P, n_cols), dtype=np.float32)
        cost[:, :A] = 1.0 - iou_pa

        row_ind, col_ind = linear_sum_assignment(cost)

        for r, c in zip(row_ind, col_ind):
            p_key = p_keys[r]
            if c >= A:
                pickupable_to_map[p_key] = None
                continue

            chosen_bbox_idx = allowed_bbox_indices[c]
            chosen_iou = float(iou_pa[r, c])

            if chosen_iou < 1e-3:
                pickupable_to_map[p_key] = None
            else:
                pickupable_to_map[p_key] = chosen_bbox_idx

        return pickupable_to_map

    def process_queries(self, queries: List[str]) -> Dict:
        """GT-based query processing without CLIP retrieval or VLM verification."""
        results: Dict[str, Dict] = {}

        for query_text in queries:
            result_entry = self._create_empty_result_entry(query_text)

            exists = self.pickupable_existence.get(query_text, False)
            map_id = self.pickupable_map_ids.get(query_text)

            if exists and map_id is not None:
                self._fill_result_entry(result_entry, map_id)

            results[query_text] = result_entry

        return results
