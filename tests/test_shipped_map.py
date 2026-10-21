import jax
import numpy as np
import pytest

from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap

LAYOUT = 153
# (pickupable, query hour) -> (most likely receptacle, belief)
EXPECTED = {
    ("AlarmClock|surface|6|11", 505.0): ("Bed|4|0|0", 0.965495),
    ("Apple|surface|8|25", 671.0): ("ArmChair|4|3|0", 0.807589),
    ("Egg|surface|8|19", 671.0): ("Sink|5|1|0", 0.999999),
    ("Pen|surface|4|6", 671.0): ("DiningTable|6|1|0", 0.510272),
    ("ToiletPaper|surface|5|9", 671.0): ("CounterTop|8|0", 0.341353),
}


@pytest.fixture(scope="module")
def object_map(data_root, compose_cfg):
    cfg = compose_cfg("agent", [f"data_root={data_root}", f"layout_num={LAYOUT}"])
    omap = PerpetuaObjectMap.load(cfg.map_path)
    omap.update_estimators(
        data_path=cfg.data_dir,
        load_week=False,
        p_m=cfg.p_m,
        p_f=cfg.p_f,
        alpha=cfg.alpha,
        decay_rate=cfg.decay_rate,
        key=jax.random.PRNGKey(cfg.seed),
        map_time=cfg.map_time,
    )
    return omap


def test_map_contents(object_map):
    assert object_map.time == 504
    assert len(object_map.get_pickupables_name()) == 15
    assert len(object_map.get_receptacles_name()) == 14


@pytest.mark.parametrize("pickupable,hour", list(EXPECTED), ids=[f"{p.split('|')[0]}@{h:g}" for p, h in EXPECTED])
def test_predictions_match_reference(object_map, pickupable, hour):
    obj = object_map.get_pickupable(pickupable)
    belief = np.asarray(obj.predict(hour)).ravel()
    receptacle, value = EXPECTED[(pickupable, hour)]
    best = int(np.argmax(belief))
    assert obj.receptacles[best] == receptacle
    np.testing.assert_allclose(belief[best], value, atol=1e-5)

