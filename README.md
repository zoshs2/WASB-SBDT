# WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking

Code & dataset repository for the paper: **[Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://arxiv.org/abs/2311.05237)**

Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa

[![arXiv](https://img.shields.io/badge/arXiv-2311.05237-00ff00.svg)](https://arxiv.org/abs/2311.05237) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

We present Widely Applicable Strong Baseline (WASB), a Sports Ball Detection and Tracking (SBDT) baseline that can be applied to wide range of sports categories :soccer: :tennis: :badminton: :volleyball: :basketball: .

https://github.com/nttcom/WASB-SBDT/assets/63090948/8889ef53-62c7-4c97-9b33-8bf386489ba1

## News

- [11/23/2023] [Our BMVC2023 proceeding](https://proceedings.bmvc2023.org/310/) is available! Thank you, BMVC2023 organizers!
- [11/23/2023] Evaluation codes of DeepBall, DeepBall-Large and BallSeg are added!
- [11/21/2023] Evaluation codes of TrackNetV2, ResTrackNetV2 and MonoTrack are added!
- [11/17/2023] Repository is released. Now it contains evaluation codes of pretrained WASB models only. Other models will be coming soon!
- [11/09/2023] Our [arXiv preprint](https://arxiv.org/abs/2311.05237) is released.

## Installation and Setup

Tested with Python3.8, CUDA11.3 on Ubuntu 18.04 (4 V100 GPUs inside). We recommend to use the [Dockerfile](./Dockerfile) provided in this repo (with ```-it``` option when running the container).

- See [GET_STARTED.md](./GET_STARTED.md) for how to get started with SBDT models.
- See [MODEL_ZOO.md](./MODEL_ZOO.md) for available model weights.

## Running evaluation on CPU

Once the detector supports CPU execution, you can run evaluation without NVIDIA GPUs by selecting the CPU preset or overriding the runner settings. The following examples evaluate the tennis dataset on CPU:

```bash
# Use the dedicated CPU config
python3 main.py --config-name=eval_cpu dataset=tennis dataloader.num_workers=2

# Or override the runner settings on the default eval config
python3 main.py --config-name=eval dataset=tennis runner.device=cpu runner.gpus=[] dataloader.num_workers=2
```

**Custom clips:** the tennis dataloader expects `<root>/<match>/<clip>/` folders that contain frames plus `Label.csv`. When using the provided preprocessing notebook, each MP4 becomes a single clip and the prepared frames are placed directly under `<root>/<match>/` with `Label.csv`; this flat layout is supported as well.

## Docker + custom tennis inference quickstart (CPU)

1. Build the image from the repo root (no GPU flags needed on Intel/macOS):
   ```bash
   docker build -t wasb-sbdt .
   ```

2. Prepare your custom clips with `notebooks/preprocess_myInput.ipynb` so you have a layout like:
   ```
   myInput_prepared/myInput/
     game1/sample1/{000000.jpg, 000001.jpg, ..., Label.csv}
     game2/sample2/{...}
     game3/sample3/{...}
     game4/sample4/{...}
   ```

3. Run the container and mount the prepared data (adjust the host path as needed):
   ```bash
   docker run --rm -it \
     -v $(pwd):/workspace/WASB-SBDT \
     -v /absolute/path/to/myInput_prepared:/workspace/myInput_prepared \
     wasb-sbdt /bin/bash
   ```

4. Inside the container, run CPU inference with overlays for the four games:
   ```bash
   cd /workspace/WASB-SBDT/src
   python3 main.py --config-name=eval_cpu \
     dataset=tennis model=wasb \
     dataset.root_dir=/workspace/myInput_prepared/myInput \
     dataset.test.matches=[game1,game2,game3,game4] \
     runner.device=cpu runner.gpus=[] \
     runner.vis_result=true \
     detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar \
     dataloader.num_workers=2
   ```
   - Use `runner.vis_hm=true` or `runner.vis_traj=true` if you also want heatmap/trajectory overlays.
   - If your match folders are named differently, update `dataset.test.matches` to match the directory names under `dataset.root_dir`.

## Citation

If you find this work useful, please consider to cite our paper:

```
@inproceedings{tarashima2023wasb,
	title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
	author={Tarashima, Shuhei and Haq, Muhammad Abdul and Wang, Yushan and Tagawa, Norio},
	booktitle={BMVC},
	year={2023}
}
```


