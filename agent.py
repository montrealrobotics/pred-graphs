import logging
import os
import sys
from pathlib import Path

import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig

sys.stdout.reconfigure(line_buffering=True)

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")

from concept_graphs.utils import set_seed, pretty_print_config

# A logger for this file
log = logging.getLogger(__name__)


def run_demo_queries(agent, cfg: DictConfig) -> None:
    from semistaticsim.keyboardcontrol.main_skillsim import ROBOTS
    from semistaticsim.rendering.ai2thor_hippo_controller import random_teleport

    # Skip blacklisted pickupables (these are bogus objects)
    blacklist = []
    blacklist_path = Path("assets/procthor/blacklist") / f"{cfg.layout_num}.yaml"
    if blacklist_path.exists():
        blacklist = yaml.safe_load(blacklist_path.read_text())
    pickupables = [p for p in agent.toolbox.object_map.get_pickupables_name() if p not in blacklist]

    key = jax.random.PRNGKey(cfg.seed + cfg.layout_num)
    for i in range(cfg.demo_queries):
        key, rng_object, rng_pose = jax.random.split(key, 3)
        pickupable_id = pickupables[int(jax.random.choice(rng_object, len(pickupables)))]
        random_teleport(rng_pose, agent.sim.controller, agent.sim.full_reachability_graph)
        if not agent.stop_server:
            obs = agent.sim.render()
            agent.sim.privileged_apn = None
            agent.server.display_agent(pose=obs["agent_pose"], img=obs["rgb_first_view"], intrinsics=obs["intrinsics"])

        log.info(f"Demo query {i + 1}/{cfg.demo_queries}: looking for '{pickupable_id}'")
        result = agent.goto_object(pickupable_id)
        actually_absent = agent.sim.SPLMetadata(ROBOTS[0], pickupable_id)[0]["object_absent"]
        outcome = "found it" if result.found else ("reported it absent" if result.is_absent else "did not find it")
        log.info(
            f"Agent {outcome} after {result.stats.get('steps', 0)} steps; "
            f"the object is {'absent from' if actually_absent else 'present in'} the scene."
        )


@hydra.main(version_base=None, config_path="conf", config_name="agent")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    pretty_print_config(cfg)
    from semistaticsim.keyboardcontrol import main_skillsim

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cfg_path = Path(cfg.data_dir).resolve() / f"run_{cfg.world_time}"
    gt_cfg_path = Path(cfg.data_dir).resolve().parent / "groundtruth"
    start_at = f"{cfg.world_time} hour"
    assert cfg.world_time >= cfg.map_time, "World time must be >= map time."

    # Instantiate simulator
    sim = main_skillsim.main(
        config_path=str(cfg_path),
        overrides=[
            "mode.runfunc=rollout",
            "mode.get_simulator_instance=true",
            "timespan_to_collect=1 hour",
            f"mode.im_width={cfg.width}",
            f"mode.im_height={cfg.height}",
            f"index={cfg.layout_num}",
            f"start_at={start_at}",
            f"mode.apn_dist_thresh={cfg.apn_dist_thresh}",
            f"scene={str(gt_cfg_path)}",
        ],
    )

    agent = instantiate(cfg.agent, sim=sim)
    if cfg.demo_queries > 0:
        run_demo_queries(agent, cfg)
    agent.spin()


if __name__ == "__main__":
    main()
