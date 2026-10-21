from typing import Optional, Self, List, Set, Literal
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Vector:
    vector: np.ndarray  # Shape (3,)
    count: int = 1
    object_ids: Set[str] = field(default_factory=set)
    last_updated_step: int = 0

    def create(vector: np.ndarray, object_id) -> Self:
        return Vector(vector=vector, object_ids={object_id})

    def merge(self, other: Self):
        """
        Updates this vector by merging another Vector into it.
        Handles weighted averaging of positions and union of object IDs.
        """
        total_count = self.count + other.count

        # Weighted Average: (PosA * CountA + PosB * CountB) / Total
        weighted_sum = (self.vector * self.count) + (other.vector * other.count)
        self.vector = weighted_sum / total_count

        # Merge counts
        self.count = total_count

        # Merge identities (Union of sets)
        self.object_ids.update(other.object_ids)
        self.last_updated_step = max(self.last_updated_step, other.last_updated_step)


@dataclass
class CanonicalVectors:
    vectors: List[Vector] = field(default_factory=list)
    merge_radius: float = 0.1
    global_step: int = 0

    def __iadd__(self, vector: Vector) -> Self:
        self.update(vector)
        return self

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, index: int) -> Vector:
        return self.vectors[index]

    def add_vectors(self, other: Self):
        for vector in other.vectors:
            self.update(vector)

    def sample(self, vectors: List[Vector] = None) -> Vector:
        if vectors is None:
            vectors = self.vectors
        # Sample based on count of each vector
        counts = np.array([v.count for v in vectors])
        probabilities = counts / counts.sum()
        chosen_index = np.random.choice(len(vectors), p=probabilities)
        return vectors[chosen_index]

    def get_vector_for_object(
        self, object_id: str, strategy: Literal["last_seen", "sample"] = "sample"
    ) -> Optional[Vector]:
        """
        Retrieves a vector for a specific object.
        :param strategy: 'last_seen' returns the most recently updated vector.
                         'sample' returns a random vector (weighted by count).
        """
        # Find all vectors containing this object
        candidates = [v for v in self.vectors if object_id in v.object_ids]

        if not candidates:
            return None

        # Strategy: Last Seen
        if strategy == "last_seen":
            # Sort by timestamp descending (newest first) and pick the first
            candidates.sort(key=lambda v: v.last_updated_step, reverse=True)
            return candidates[0]

        # Strategy: Sample
        elif strategy == "sample":
            return self.sample(candidates)

        return None

    def update(self, new_input: Vector):
        self.global_step += 1
        # Update new input timestamp
        new_input.last_updated_step = self.global_step

        best_match_idx = -1
        for i, existing_vec in enumerate(self.vectors):
            dist = np.linalg.norm(existing_vec.vector - new_input.vector)
            if dist < self.merge_radius:
                best_match_idx = i
                break

        if best_match_idx != -1:
            # Merge
            self.vectors[best_match_idx].merge(new_input)
        else:
            # Create new
            self.vectors.append(new_input)