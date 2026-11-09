# DisasterAI — runnable flood-segmentation demo

This project is prepared for a Windows + VS Code demonstration. It contains three **separate experiments**:

1. **Existing/traditional baseline:** SAR backscatter intensity thresholding (validation-selected threshold, implemented locally).
2. **Existing/deep-learning baseline:** a vanilla U-Net trained from random initialization.
3. **Proposed method:** a custom attention-gated U-Net trained from random initialization, with input-gradient saliency visualization.

No pretrained weights, pretrained CNNs, pretrained segmentation networks, BERT, or pretrained embeddings are used.

## Important scientific scope

- Sen1Floods11 contains different subsets/label types. Some public examples use permanent-water labels, which are **not the same as flood-event labels**.
- Use hand-labeled flood-event imagery/masks for a flood segmentation claim. If you use the permanent-water subset, describe the experiment as surface/permanent-water segmentation.
- Do not present any metric in this README as a measured result. Metrics are created only when you run the scripts.
- `--demo-synthetic` generates artificial toy images solely to test that the code executes. Synthetic-demo scores are not evidence of disaster detection and must not be presented as real results.
- Otsu thresholding on SAR backscatter is a simple reference baseline; it is not guaranteed to perform well on all scenes.
- The proposed attention U-Net is a prototype architecture. Whether it improves on vanilla U-Net must be established experimentally.

## 1. Setup in Windows / VS Code

Use Python 3.10–3.12 in a virtual environment. In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For an Intel Core Ultra integrated GPU, first update/install the Intel graphics driver, then install PyTorch's Intel XPU wheels using the official PyTorch instructions. The current documented stable command is:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

Official instructions: https://docs.pytorch.org/docs/main/notes/get_start_xpu.html

Then install the remaining requirements:

```powershell
pip install -r requirements.txt
```

The script automatically chooses CUDA, then Intel XPU, then CPU. Intel GPU support depends on your exact Core Ultra generation, Windows version, driver, and installed wheel. If XPU is unavailable, the script falls back to CPU; do not assume GPU acceleration until the check below says `xpu available: True`.

Check GPU access:

```powershell
python -c "import torch; print('torch:',torch.__version__); print('cuda:',torch.cuda.is_available()); print('xpu available:',hasattr(torch,'xpu') and torch.xpu.is_available()); print('xpu name:',torch.xpu.get_device_name(0) if hasattr(torch,'xpu') and torch.xpu.is_available() else 'not available')"
```

## 2. Get the real dataset

Primary candidate: Sen1Floods11, official repository and paper:
- https://github.com/cloudtostreet/Sen1Floods11
- https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_Flood_Algorithms_CVPRW_2020_paper.html

Download the **hand-labeled flood-event** SAR images and matching label masks according to the official dataset documentation. The original dataset is hosted in public Google Cloud Storage; download instructions may use `gsutil` or other Google Cloud tooling. If using the `flood_handlabeled` split, preserve its official train/validation/test assignments where possible.

Create this local layout (GeoTIFF or PNG masks are supported):

```text
data/
  images/
    event001.tif
    event002.tif
  masks/
    event001.tif
    event002.tif
  splits/
    train.txt
    val.txt
    test.txt
```

`images` and `masks` must have matching file stems. A split file contains one stem per line, for example `event001`. Use the official split CSVs to create these text files; do not randomly split chips from the same event if the official split is available. For this demo loader, masks should use 0 for background, positive values for the target class, and negative/ignore labels will be excluded where present. Check label encoding before training.

The loader reads GeoTIFF via Rasterio. For multi-band Sentinel-1 inputs it takes the first two bands (VV and VH if that is how your files are ordered), clips each band to [-50, 1] dB, and scales to [0, 1]. Verify band order and value range in your downloaded data before using the default preprocessing.

If your local files are not named with matching stems, make a manifest CSV with columns `image_path,mask_path,split` and adapt the file pairs accordingly.

## 3. Verify the project runs immediately (synthetic smoke test)

This does not require a dataset. It only verifies code execution:

```powershell
python run_experiment.py --demo-synthetic --epochs 1 --max-train 12 --max-val 4 --max-test 4
```

The script writes outputs under `outputs/`. Synthetic scores are strictly smoke-test output, not valid project results.

## 4. Run the three methods on your real dataset

First inspect the dataset and confirm the image/mask pairing and labels. Then run:

```powershell
python run_experiment.py --data-root data --epochs 5 --batch-size 4 --image-size 128
```

For a quick GPU check before the full run, use:

```powershell
python run_experiment.py --data-root data --epochs 1 --max-train 16 --max-val 8 --max-test 8 --batch-size 2 --image-size 128
```

If your machine has enough GPU memory, increase image size to 256 and train longer. A short run is a smoke test, not necessarily a reliable final evaluation.

## 5. What gets saved

The script creates:
- `outputs/metrics.json` — measured metrics from the run
- `outputs/metrics.csv` — comparison table
- `outputs/training_curves.png` — loss curves for learned models
- `outputs/predictions.png` — input, ground truth, and predictions
- `outputs/threshold_predictions.png` — threshold baseline visualizations
- `outputs/checkpoints/unet_best.pt`
- `outputs/checkpoints/proposed_attention_unet_best.pt`
- `outputs/run_config.json` — settings and device used

If a metric is missing or a method did not finish, report it as missing/failed, not as zero.

## 6. Method definitions

### Method 1 — Traditional SAR threshold baseline
Intensity thresholding of a selected SAR channel; threshold and polarity are selected on validation data. This is a conventional non-neural reference method, inspired by threshold-based water-mapping baselines. Depending on the data convention, darker SAR backscatter can indicate water, but a threshold must be selected on validation data and the direction must be checked. The script selects the better polarity on validation only.

### Method 2 — Vanilla U-Net
A small encoder-decoder CNN with skip connections, trained from random initialization. No pretrained weights.

### Method 3 — Proposed attention-gated U-Net
The same general segmentation task, but with attention gates on skip features and a simple input-attribution map. The proposed method is not claimed to be better until the measured test comparison supports that claim.

## 7. Metrics

The script reports pixel-level IoU, Dice, precision, recall, and pixel accuracy, ignoring pixels labeled as negative. These metrics can be misleading if masks have incorrect encodings or strong class imbalance. Inspect predictions and label values before interpreting the scores.

## 8. Tomorrow's presentation (2–3 minutes)

1. State the problem: disaster mapping needs interpretable segmentation and robust evaluation.
2. Show the data source, one SAR image, and its human-labeled mask.
3. Run/show the threshold baseline and explain its limitations.
4. Run/show the vanilla U-Net, trained from scratch.
5. Run/show the proposed attention U-Net and its input-gradient saliency visualization.
6. Show the automatically generated comparison table. Quote only metrics from your own `outputs/metrics.csv`.
7. Explain the limitation: attention U-Net is a proposed variant; improvement is not assumed. Mention that full multimodal fusion (imagery + environmental time series + text) is future integration work unless you have implemented and tested it.

## References

- Bonafilia et al. (2020), *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms*, CVPR Workshops.
- Official dataset/code: https://github.com/cloudtostreet/Sen1Floods11
- PyTorch installation: https://pytorch.org/get-started/locally/
