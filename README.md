# JOPAT

**JOPAT: Joint Observation–Action Prediction with Auxiliary Tracks**

This repository contains the anonymous reference implementation of JOPAT for
double-blind review. JOPAT trains one diffusion transformer to jointly predict
an action chunk, future visual observations, and dense point trajectories.
Current observations provide global conditioning, while auxiliary observation
and track objectives supply predictive supervision during training.

The release contains source code and portable configurations only. Datasets,
checkpoints, generated media, experiment logs, machine-specific launch files,
and identifying metadata are intentionally excluded.

## Repository layout

```text
configs/                 Hydra configurations
datasets/                LIBERO/robomimic and DROID data pipelines
environments/            Rollout wrappers
experiments/jopat/       Training, validation, and rollout entry points
models/common/           Shared neural-network components
models/jopat/            JOPAT model and checkpoint utilities
tests/                   Lightweight model smoke tests
```

## Requirements

The reference environment uses Python 3.10, PyTorch 2.2, and CUDA. A GPU is
required for training and for generating dense tracks. Linux is recommended
for MuJoCo-based evaluation.

JOPAT uses three external projects:

- [CoTracker](https://github.com/facebookresearch/co-tracker) for track
  generation.
- [robomimic](https://github.com/ARISE-Initiative/robomimic) for dataset and
  environment utilities.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for the LIBERO
  benchmark.

Create an isolated environment:

```bash
conda create -n jopat python=3.10 -y
conda activate jopat
python -m pip install --upgrade pip
```

Install the external projects in a separate `third_party` directory. The
CoTracker revision below matches the implementation used by this release.

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/co-tracker.git third_party/co-tracker
git -C third_party/co-tracker checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
python -m pip install -e third_party/co-tracker

git clone https://github.com/ARISE-Initiative/robomimic.git third_party/robomimic
git -C third_party/robomimic checkout 0ca7ce74cf8f20be32029657ef9320db033d93e9
python -m pip install -e third_party/robomimic

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 2a3b44044398c5b2a1dc4904fa3a01269ebc7bc0
python -m pip install -r third_party/LIBERO/requirements.txt
python -m pip install -e third_party/LIBERO

python -m pip install -e .
```

Install JOPAT last so that its tested diffusion and transformer versions take
precedence over older constraints declared by external packages.

Follow the CoTracker documentation to obtain the `scaled_offline` checkpoint.
Follow the LIBERO and DROID instructions to download their public datasets.
The first model construction also downloads the public
`stabilityai/sdxl-vae` weights and the ImageNet ResNet-18 weights. Set
`HF_HOME` and `TORCH_HOME` if these caches must live outside their default
locations.

## LIBERO data

The LIBERO configuration reads all machine-dependent locations from environment
variables:

```bash
export LIBERO_HDF5_GLOB='/path/to/libero/datasets/libero_90/*.hdf5'
export COTRACKER_CHECKPOINT='/path/to/scaled_offline.pth'
export JOPAT_CACHE_DIR='/path/to/jopat_cache'
mkdir -p "$JOPAT_CACHE_DIR"
```

On the first run, the dataset pipeline converts the HDF5 demonstrations to
compressed Zarr storage and generates window-aligned CoTracker trajectories.
Later runs restore these caches. Track generation is compute- and
storage-intensive; set `dataset.cotracker_batch_size` to fit the available GPU.

To use a single LIBERO task, point `LIBERO_HDF5_GLOB` to that task's HDF5 file.
To use another suite, point it to the corresponding glob. Use a distinct
`JOPAT_CACHE_DIR` for every dataset or task; an existing cache is restored as-is
and must not be reused after changing `LIBERO_HDF5_GLOB`. No source-code change
is required.

## Training

Run JOPAT with the default LIBERO configuration:

```bash
python -m experiments.jopat.train --config-name train
```

The launcher uses every visible CUDA device and starts one distributed worker
per device. Restrict the run to selected devices with `CUDA_VISIBLE_DEVICES`.
Hydra overrides can be supplied on the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.jopat.train \
  --config-name train \
  batch_size=16 \
  num_workers=4 \
  exp_id=libero90
```

Outputs are written below `outputs/JOPAT/<dataset>/<experiment>/<seed>/`.
Weights & Biases logging is disabled by default. Enable it only in a
non-anonymous environment:

```bash
python -m experiments.jopat.train --config-name train \
  wandb.enabled=true \
  wandb.project=JOPAT
```

Do not place API keys, account names, or team identifiers in configuration
files. Use the standard local W&B login mechanism when logging is enabled.

## Fine-tuning and resuming

Set the pretraining checkpoint through an environment variable and select the
downstream task through `LIBERO_HDF5_GLOB`:

```bash
export JOPAT_PRETRAIN_CHECKPOINT='/path/to/pretrained_jopat.pt'
export LIBERO_HDF5_GLOB='/path/to/downstream_task_demo.hdf5'
export JOPAT_CACHE_DIR='/path/to/jopat_cache/downstream_task'

python -m experiments.jopat.train --config-name finetune \
  exp_id=downstream_task
```

Resume an interrupted run with:

```bash
python -m experiments.jopat.train --config-name train \
  resume=true \
  resume_checkpoint_path='/path/to/checkpoint.pt'
```

Checkpoints contain the model, optimizer, scheduler, scaler, and training step.
They do not require W&B to be enabled.

## DROID data

Convert a public DROID TFDS dataset into the compressed episode buffers:

```bash
python -m datasets.droid.convert_dataset_zarr \
  --data_dir /path/to/tfds \
  --buffer_path_action /path/to/droid_action.zarr \
  --num_action_episodes 2000
```

To create an additional zero-action buffer, also pass
`--buffer_path_action_free` and a positive `--num_action_free_episodes`.
Training consumes the buffer selected by `DROID_BUFFER_PATH`.

Then configure the track cache and train:

```bash
export DROID_BUFFER_PATH='/path/to/droid_action.zarr'
export COTRACKER_CHECKPOINT='/path/to/scaled_offline.pth'
export JOPAT_CACHE_DIR='/path/to/jopat_cache'

python -m experiments.jopat.train --config-name train dataset=droid_track
```

## Evaluation

Validation losses are computed every `eval_every` steps. For a single LIBERO
task, environment rollouts are computed every `rollout_every` steps and at the
end of training. Set `num_rollouts` and `rollout_length` to match the evaluation
protocol. Multi-task suites should be evaluated one task at a time by changing
`LIBERO_HDF5_GLOB`; this keeps task initialization and success accounting
unambiguous.

Evaluate a compatible checkpoint independently:

```bash
export JOPAT_CHECKPOINT='/path/to/jopat_checkpoint.pt'
export LIBERO_HDF5_GLOB='/path/to/single_task_demo.hdf5'

python -m experiments.jopat.evaluate
```

## Configuration notes

- `num_frames=21` is the training sequence length: `o_{t-1:t}`, the action chunk, the two future observations, and the track target.
- `obs_num_frames=2` is the observation horizon; the last frame is the tracking reference.
- `model.action_len=19` is the predicted action horizon.
- `model.track_len=19` is the track horizon, covering frames `t+1` through `t+H_p`; the reference frame is excluded.
- `model.future_offset=16` is the offset of the predicted future observation window.
- `model.num_points=625` corresponds to a 25 × 25 query grid.
- `model.num_registers=8` is the default transformer register count.
- `model.use_track_visibility=true` enables the visibility objective.
- Image and track paths are runtime inputs and must not be committed.

All commands are intended to run from the repository root. To inspect the fully
resolved configuration without starting a run:

```bash
python -m experiments.jopat.train --config-name train --cfg job --resolve
```

Run the architecture smoke tests without datasets or pretrained weights:

```bash
python -m unittest discover -s tests -v
```

## Anonymous-release policy

The repository contains no author list, affiliations, personal URLs, private
storage paths, cluster accounts, service credentials, experiment dashboards,
or identifying development history. Citation metadata and authorship
information will be added only after the double-blind review period.
