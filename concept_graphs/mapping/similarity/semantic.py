import torch
from .Similarity import SemanticSimilarity


class CosineSimilarity01(SemanticSimilarity):
    def __call__(
        self, main_semantic: torch.Tensor, other_semantic: torch.Tensor
    ) -> torch.Tensor:
        return main_semantic @ other_semantic.t() / 2 + 0.5
