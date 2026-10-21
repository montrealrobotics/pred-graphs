# Creating your own data

This guide goes from a ProcTHOR house to a Predictive Graph you can run the [navigation demo](README.md#navigation-demo) on:

1. [Generate SemiStaticSim data](#1-generate-semistaticsim-data): object schedules and the robot's hourly runs.
2. [Fit Perpetua* estimators](#2-fit-perpetua-estimators) for every object.
3. [Build the map](#3-build-the-map) from a few runs before the map time.
4. [Run the demo](#4-run-the-demo) on it.

Commands run from the repository root unless stated otherwise. `<layout>` is a ProcTHOR-10k test-split house index; the example data uses 153.

## Data layout

```
data/
├── object_data/test/<layout>/   # step 1a: objects extracted from the ProcTHOR house
├── semistaticsim/test/<layout>/0/
│   ├── groundtruth/             # step 1a: object schedules
│   └── privileged/run_<hour>/   # step 1b: one robot run per hour
├── estimators/scene_<layout>/   # step 2
└── outputs/latest_map           # step 3
```

Keep the `test/<layout>/0` part of the path: SemiStaticSim reads the split, layout and seed from it.

## 1. Generate SemiStaticSim data

[SemiStaticSim](https://github.com/Velythyl/SemiStaticSim) simulates a house over four weeks (672 hours). It first samples where each small object is at every hour (the ground truth), then renders a robot exploring the house each hour, recording RGB-D frames, poses, semantic masks and the objects it saw (the privileged runs).

### 1a. Object schedules

```bash
uv run python -m semistaticsim.groundtruth.main procthor_index=<layout> data_dir=$PWD/data
```

This writes the schedules to `data/semistaticsim/test/<layout>/0/groundtruth/`. The first run of a house downloads ProcTHOR-10k and extracts the house's objects to `data/object_data/`. The default configuration is the one of the example data: 15 pickupables (`objects.limit_num_pickupables`), at most 3 receptacles each, a weekly and daily routine (`schedule`), and a 10% chance for an object to leave the house (`objects.oob_receptacle_prob`). Override them on the command line.

### 1b. Robot runs

```bash
uv run python -m semistaticsim.keyboardcontrol.main_skillsim mode=auto index=<layout> data_dir=$PWD/data start_at="0 hour" timespan_to_collect="14 hour"
```

This writes `run_0` ... `run_13` to `data/semistaticsim/test/<layout>/0/privileged/`. Each hour takes about 10 minutes on an NVIDIA L40S and 120 MB of disk, so all 672 hours take roughly 120 GPU-hours and 80 GB. Chunks (`start_at="0 hour"`, `"14 hour"`, ..., `"658 hour"`) can run in parallel on several GPUs.

What each later step needs:

| Step           | Runs needed                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| Fit estimators | All 672 hours, only `*.parquet`, `*.json` and `config.yaml` (images can be deleted) |
| Build the map  | RGB-D for a few hours before the map time (the example uses 485-504)                |
| Demo           | `config.yaml` of the hours the simulator starts at (`world_time`)                   |

## 2. Fit Perpetua* estimators

```bash
JAX_PLATFORMS=cpu uv run python perpetua2/entrypoint.py --config-name train_procthor run_id=<layout> sequence_path=data/semistaticsim/test/<layout>/0/privileged output_dir=data/estimators/scene_<layout> results_dir=data/outputs/estimators/scene_<layout>
```

This fits one Perpetua* estimator per pickupable on the first 504 hours and plots its predictions on the last 168. It takes about 3 hours on 4 CPU cores and writes one `<object>_object.pkl` per pickupable to `data/estimators/scene_<layout>/`, with plots in `results_dir`. See the [Perpetua* README](https://github.com/montrealrobotics/perpetua-star) for the model.

## 3. Build the map

```bash
uv run python orchestrate.py layout_num=<layout> keyframes=true
```

This builds an object map from the runs just before hour 504 (`first_run_index`), matches its objects to the house's pickupables and receptacles, and attaches the estimators from step 2. Segmentation uses SAM3 by default (see [Perception](README.md#perception)); `segmentation_model=GTProcthor` uses the simulator's segmentation and needs no model. The map is written to `data/outputs/scene_<layout>_.../`, and `data/outputs/latest_map` points to it.

To build the map at another time, set `first_run_index=<hour>` and use the same value for `map_time` in the next step.

## 4. Run the demo

```bash
uv run python agent.py layout_num=<layout> map_path=data/outputs/latest_map demo_queries=3
```

`world_time` (524 by default) must be at least the map time. To keep some pickupables out of the demo queries, list them in `assets/procthor/blacklist/<layout>.yaml`.
