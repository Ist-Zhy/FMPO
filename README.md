# FMPO

Research code for infrared–visible image fusion with flow matching and reward-guided optimization. This release includes Stage I and Stage II training, and the post-training TensorRT quantization scripts. Model weights (including external reward weights), datasets, generated ONNX files and TensorRT engines are not included.

## Environment

The experiments used Python 3.10, PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1 and an NVIDIA RTX 3090. Install a compatible [PyTorch](https://pytorch.org/get-started/previous-versions/) build for your CUDA environment, then `pip install -r requirements.txt`. Stage II additionally needs an external [DINOv3](https://github.com/facebookresearch/dinov3) source checkout and ViT-B/16 pretrained checkpoint, and OpenCLIP ViT-B-16 weights. Install the dependencies required by that DINOv3 checkout and obtain those weights separately under their respective licenses.

Quantization requires ONNX 1.17.0 and TensorRT 10.8.0.43 with matching CUDA libraries. TensorRT engines are GPU- and environment-specific and must be built locally.

## Training

Run commands from the repository root. Edit the YAML files to point to local data and checkpoints. By default both stages use width 64; when changing width, change `Diffusion.dim` in both configurations and use a matching checkpoint.

Stage I expects paired source images and pseudo-fused targets with identical filenames:

```
data/stage1/
  Train/{A_N,B_N,Fusion}/
  Val/{A_N,B_N,Fusion}/
```

`A_N` is infrared and `B_N` is visible. Stage I samples aligned random crops; input images must be at least the configured crop size. The validation `Fusion` folder may be omitted. Train with:

```bash
python Flow_Train.py --config configs/Flow_Train.yaml
```

Stage II expects already prepared paired patches in `data/stage2/Train/{ir,vi}/` and validation images in `data/stage2/Val/{ir,vi}/`; matching `fu`/`Fusion` folders are optional. Set `RL_parameters.sft_checkpoint_path` to the Stage I checkpoint. Set the reward paths before running:

```bash
export FMPO_DINOV3_REPO=/path/to/dinov3
export FMPO_DINOV3_WEIGHTS=/path/to/dinov3_vitb16.pth
export FMPO_CLIP_WEIGHTS=/path/to/open_clip_pytorch_model.bin
python FMPO_Test.py --config configs/Flow_Train_RL_plus.yaml --gpu 0
```

`FMPO_Test.py` is the renamed Stage II training entry point. Both rewards must load successfully. The Stage II checkpoint loader requires an exact architecture and width match. The exact training data preparation, third-party pretrained weights, and full GPU training were not validated in this release environment; results can depend on those inputs and software versions.

## TensorRT quantization

The pipeline loads a Stage II checkpoint strictly, merges LoRA if present, folds weight-standardized convolutions, verifies one-step conversion parity, calibrates on training pairs, exports ONNX, and builds FP32, FP16 and mixed INT8/FP16 TensorRT engines. INT8 uses per-channel weights and per-tensor activation scales for convolutions and linear layers; other operations can remain floating point.

Provide your own checkpoint and calibration image folders. Each calibration directory contains `ir/` and `vi/` subdirectories, matched by filename stem. Defaults are `data/{MSRS,FMB}/train/{ir,vi}`.

```bash
export FMPO_CHECKPOINT_64=/path/to/rl_final.pt
export FMPO_MSRS_CALIBRATION=/path/to/msrs/train
export FMPO_FMB_CALIBRATION=/path/to/fmb/train
python experiments/quantization/run_quantization.py prepare --dim 64
for mode in fp32 fp16 int8; do
  python experiments/quantization/run_quantization.py build --dim 64 --mode "$mode"
done
```

For widths 16 or 32 set `FMPO_CHECKPOINT_16` or `FMPO_CHECKPOINT_32`, and run prepare and build separately for each width. Generated files are saved under `results_quantization/dim64/` (or the corresponding width). ONNX files and engines contain model parameters and are not included in this repository.

Optionally set `FMPO_MSRS_IR`, `FMPO_MSRS_VI`, `FMPO_FMB_IR`, and `FMPO_FMB_VI` to test image directories to check for overlap with calibration pairs. If the corresponding test directories are not configured or present under `data/{MSRS,FMB}/test/{ir,vi}`, the overlap check is skipped and the script reports that fact. Use training images for calibration.
