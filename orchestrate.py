from colorama import Fore, Style
import re
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import hydra
from omegaconf import DictConfig
import dill as pickle

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from concept_graphs import utils

log = logging.getLogger(__name__)


def get_available_scenes(data_dir: Path, first_run_index: int = 0) -> List[Tuple[int, str]]:
    scenes = []

    if not data_dir.exists():
        log.warning(f"Data directory does not exist: {data_dir}")
        return scenes

    for d in data_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            match = re.match(r"run_(\d+)", d.name)
            if match:
                run_num = int(match.group(1))
                scenes.append((run_num, d.name))

    # Sort by run number descending (latest first)
    scenes.sort(key=lambda x: x[0], reverse=True)

    if first_run_index is not None and first_run_index > 0 :
        scenes = [s for s in scenes if s[0] <= first_run_index]

    return scenes


def run_main_py(
    scene: str,
    sequence_stride: int,
    keyframes: bool,
    main_config_name: str,
    layout_num: int,
    output_dir: Path,
    overrides: List[str],
) -> subprocess.CompletedProcess:
    """
    Run main.py to create ObjectMap for a scene.
    """
    cmd = [
        sys.executable,
        "main.py",
        f"--config-name={main_config_name}",
        f"layout_num={layout_num}",
        f"output_dir={output_dir}",
        f"dataset.scene={scene}",
        f"dataset.keyframes={keyframes}",
        f"dataset.sequence_stride={sequence_stride}",
        *overrides,
    ]

    log.info(f"Running main.py for scene {scene}")
    log.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result


def run_map_query_py(
    map_path: Path,
    scene: str,
    full_overwrite: bool,
    query_config_name: str,
    layout_num: int,
    output_dir: Path,
    overrides: List[str],
) -> subprocess.CompletedProcess:
    """
    Run map_query.py to update PerpetuaObjectMap.
    """
    cmd = [
        sys.executable,
        "map_query.py",
        f"--config-name={query_config_name}",
        f"layout_num={layout_num}",
        f"map_path={map_path}",
        f"dataset.scene={scene}",
        f"perpetua_map_full_overwrite={str(full_overwrite).lower()}",
        f"output_dir={output_dir}",
        *overrides,
    ]

    log.info(f"Running map_query.py on map at {map_path}")
    log.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    return result


def check_map_complete(map_path: Path) -> Tuple[bool, Optional[object]]:
    """
    Check if the PerpetuaObjectMap at the given path is complete.
    """
    from concept_graphs.mapping.PerpetuaObjectMap import MapState
    perpetua_map_file = map_path / "perpetua_map.pkl"

    if not perpetua_map_file.exists():
        log.warning(f"PerpetuaObjectMap file does not exist: {perpetua_map_file}")
        return False, None

    try:
        with open(perpetua_map_file, "rb") as f:
            perpetua_map = pickle.load(f)

        is_complete = perpetua_map.state == MapState.COMPLETE

        # Log detailed status
        n_pickupables = len(perpetua_map._pickupables)
        n_receptacles = len(perpetua_map._receptacles)
        target_pickupables = len(perpetua_map.pickupable_names)
        target_receptacles = len(perpetua_map.receptacle_names)

        log.info(
            f"{Fore.GREEN}PerpetuaObjectMap status: {perpetua_map.state.value} "
            f"(pickupables: {n_pickupables}/{target_pickupables}, "
            f"receptacles: {n_receptacles}/{target_receptacles}){Style.RESET_ALL}"
        )

        return is_complete, perpetua_map
    except Exception as e:
        log.error(f"Failed to load PerpetuaObjectMap: {e}")
        return False, None


def _main_impl(cfg: DictConfig):
    """Main orchestration loop."""

    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)

    log.info(f"Data directory: {data_dir}")
    log.info(f"Output directory: {output_dir}")
    path_overrides = [f"{k}={cfg[k]}" for k in ("cache_dir", "data_dir", "estimators_dir") if k in cfg]

    # Get available scenes
    scenes = get_available_scenes(data_dir, first_run_index=cfg.first_run_index)
    if not scenes:
        log.error(f"No scenes found in {data_dir}")
        return

    # Limit iterations
    max_iterations = min(cfg.max_iterations, len(scenes))
    log.info(f"Will process up to {max_iterations} scenes")

    prev_map_path: Optional[Path] = None

    for i, (run_num, scene_name) in enumerate(scenes[:max_iterations]):
        log.info(f"\n{'='*60}")
        log.info(f"Processing scene {scene_name} ({i+1}/{max_iterations})")
        log.info(f"{'='*60}")

        # Step 1: Run main.py to create ObjectMap
        result = run_main_py(
            scene=scene_name,
            layout_num=cfg.layout_num,
            sequence_stride=cfg.sequence_stride,
            keyframes=cfg.keyframes,
            main_config_name=cfg.main_config_name,
            output_dir=output_dir,
            overrides=path_overrides + [f"segmentation={cfg.segmentation_model}"],
        )

        if result.returncode != 0:
            log.error(f"main.py failed for scene {scene_name}")
            continue

        # Step 2: Get the latest map path
        current_map_path = utils.get_latest_map_path(output_dir)

        if current_map_path is None:
            log.error(f"Could not find latest_map symlink after running main.py")
            continue

        log.info(f"Current map path: {current_map_path}")

        # Step 3: Copy perpetua_map.pkl from previous map if exists
        if prev_map_path is not None:
            utils.copy_perpetua_map(prev_map_path, current_map_path)

        # Step 4: Run map_query.py
        # Use full_overwrite=True only for the first iteration (no previous map)
        full_overwrite = prev_map_path is None

        result = run_map_query_py(
            map_path=current_map_path,
            layout_num=cfg.layout_num,
            scene=scene_name,
            full_overwrite=full_overwrite,
            query_config_name=cfg.query_config_name,
            output_dir=output_dir,
            overrides=path_overrides + [f"use_vlm_verification={cfg.use_vlm_verification}"],
        )

        if result.returncode != 0:
            log.error(f"map_query.py failed for scene {scene_name}")
            # Still update prev_map_path to try next scene
            prev_map_path = current_map_path
            continue

        # Step 5: Check if PerpetuaObjectMap is complete
        is_complete, perpetua_map = check_map_complete(current_map_path)

        # Update prev_map_path for next iteration
        prev_map_path = current_map_path

        if is_complete:
            log.info(f"\n{'='*60}")
            log.info(
                f"{Fore.BLUE}SUCCESS: PerpetuaObjectMap is COMPLETE after processing {scene_name}{Style.RESET_ALL}"
            )
            log.info(f"Final map saved at: {current_map_path}")
            log.info(f"{'='*60}")
            return

    # If we get here, the map is still incomplete
    log.warning(f"\n{'='*60}")
    log.warning(
        f"PerpetuaObjectMap is still INCOMPLETE after processing {max_iterations} scenes"
    )
    if prev_map_path:
        log.warning(f"Latest map saved at: {prev_map_path}")
    log.warning(f"{'='*60}")


@hydra.main(version_base=None, config_path="conf", config_name="orchestrate")
def cli_main(cfg):
    return _main_impl(cfg)


if __name__ == "__main__":
    cli_main()
