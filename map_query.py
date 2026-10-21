from pathlib import Path
import os

# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from concept_graphs.utils import set_seed

import hydra
from omegaconf import DictConfig
import logging

from concept_graphs.utils import fetch_run_id
from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from concept_graphs.inference.PerpetuaMapBuilder import PerpetuaMapBuilder

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    """Main map query function."""
    set_seed(cfg.seed)

    # Instantiate hydra objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction, device=cfg.device)
    verifier = hydra.utils.instantiate(cfg.vlm_verifier)
    semantic_sim = hydra.utils.instantiate(cfg.semantic_similarity)

    # Get inputs
    pickupable_names = dataset.get_pickupable_names()
    receptacle_names = dataset.get_receptacles_names()
    p2r_map = dataset.get_pickupable_to_receptacles()

    concept_nodes_map_path = Path(cfg.map_path)

    # Initialize Engine via Hydra
    engine = hydra.utils.instantiate(
        cfg.inference,
        ft_extractor=ft_extractor,
        semantic_sim=semantic_sim,
        verifier=verifier,
        receptacles_bbox=dataset.get_receptacles_bbox(),
        pickupable_bbox=dataset.get_pickupables_bbox(),
        pickupable_existence=dataset.get_assignment(),
        pickupable_to_receptacles=p2r_map,
    )

    # Run Query Logic
    results = engine.process_queries(queries=pickupable_names)

    # Save Results
    output_path = concept_nodes_map_path / "assignments"
    engine.save_results(results, output_path=output_path)
    log.info(f"Results saved to {output_path}")

    if not cfg.build_perpetua_map:
        log.info("Skipping PerpetuaMap build (build_perpetua_map=false)")
        return

    # Update Perpetua Map using PerpetuaMapBuilder
    pickupable_map_ids, receptacle_map_ids = engine.get_map_object_ids()
    concept_nodes_map = ObjectMap.load(concept_nodes_map_path)
    current_p2r_mapping = engine.parse_assignments(results, pickupable_names)
    map_creation_time = fetch_run_id(cfg)

    # Create PerpetuaMapBuilder and build the map
    builder = PerpetuaMapBuilder(
        estimators_path=cfg.perpetua_estimators_path,
        output_path=str(concept_nodes_map_path),
        full_overwrite=cfg.perpetua_map_full_overwrite,
    )

    perpetua_map: PerpetuaObjectMap = builder.build_or_update(
        time=map_creation_time,
        concept_nodes_map=concept_nodes_map,
        pickupable_names=pickupable_names,
        receptacle_names=receptacle_names,
        pickupable_map_ids=pickupable_map_ids,
        receptacle_map_ids=receptacle_map_ids,
        current_p2r_mapping=current_p2r_mapping,
    )
    log.info(f"PerpetuaMap saved to {concept_nodes_map_path}")

    if cfg.debug:
        # Clean a bit of memory before visualization
        del perpetua_map
        del concept_nodes_map
        log.info("Visualizing map objects...")
        engine.visualize(concept_nodes_map_path, results)


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def hydra_main(cfg: DictConfig) -> None:
    """Hydra CLI entry point."""
    main(cfg)


if __name__ == "__main__":
    hydra_main()
