import importlib
import pkgutil

import pytest
from hydra.utils import get_object
from omegaconf import OmegaConf

import concept_graphs

CONFIGS = [
    ("agent", []),
    ("agent", ["demo_queries=3", "do_update=true"]),
    ("agent", ["inference/agent@agent=ProcTHORAgent"]),
    ("main", []),
    ("main", ["segmentation=YoloMobileSAM"]),
    ("main", ["segmentation=GTProcthor"]),
    ("map_query", []),
    ("map_query", ["algo=VerifyObjectsGT"]),
    ("orchestrate", []),
    ("orchestrate", ["segmentation_model=GTProcthor", "use_vlm_verification=true"]),
    ("map_server", []),
]


def targets(node):
    if OmegaConf.is_dict(node):
        for k, v in node.items_ex(resolve=False):
            if k == "_target_":
                yield v
            else:
                yield from targets(v)
    elif OmegaConf.is_list(node):
        for v in node:
            yield from targets(v)


@pytest.mark.parametrize("module", [m.name for m in pkgutil.walk_packages(concept_graphs.__path__, "concept_graphs.")])
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("name,overrides", CONFIGS, ids=[" ".join([n, *o]) for n, o in CONFIGS])
def test_config_composes_and_targets_resolve(compose_cfg, name, overrides):
    cfg = compose_cfg(name, ["dataset.scene=run_504", *overrides] if name in ("main", "map_query") else overrides)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    for target in targets(cfg):
        get_object(target)


def test_agent_flags_reach_the_agent(compose_cfg):
    cfg = compose_cfg("agent", ["do_update=true", "step_every=1", "inference/agent@agent=ProcTHORAgent"])
    assert cfg.agent.do_update and cfg.agent.step_every == 1
