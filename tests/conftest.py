import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[1]
DATA_ENV = "PG_DATA_ROOT"


@pytest.fixture(scope="session")
def compose_cfg():
    def _compose(name, overrides=()):
        with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
            return compose(config_name=name, overrides=list(overrides))

    return _compose


@pytest.fixture(scope="session")
def data_root() -> Path:
    path = os.environ.get(DATA_ENV)
    if not path:
        pytest.skip(f"set {DATA_ENV} to the skeleton dataset root (the folder with semistaticsim/, estimators/, maps/)")
    return Path(path)
