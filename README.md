# A Standardized and Reproducible Pipeline for Fast Multi-View DRR Generation

A modular Python pipeline for generating **multi-view Digitally Reconstructed Radiographs (DRRs)** from CT volumes. It standardizes preprocessing (reorientation, isotropic resampling, center alignment, HU windowing) and performs GPU-accelerated forward projection using the [TIGRE](https://github.com/CERN/TIGRE) cone-/parallel-beam model. An optional rigid coregistration module is provided for anatomically homogeneous datasets.

> Repository: https://github.com/Istiaaak/Pipeline-for-fast-multi-view-DRR-generation

---

## Repository structure

The pipeline is split into small, single-responsibility modules. You typically only edit `config.py`.

| File | Role |
|------|------|
| `config.py` | **Central configuration.** All user-facing parameters live here (paths, geometry, HU window, detector, mode). |
| `preprocessing.py` | Loads a CT, reorients, resamples to isotropic spacing, pads/crops to a fixed matrix, and clamps to the HU window. |
| `projection.py` | Builds the TIGRE geometry from the volume and runs the forward projector to produce DRRs. |
| `coreg.py` | *Optional.* Rigid (6-DOF) coregistration to a reference volume via multi-resolution mutual information. Standalone — not called by `main.py`. |
| `utils.py` | I/O helpers: save NIfTI, save PNG, write per-case metadata. |
| `main.py` | Entry point. Iterates over input CTs, runs preprocessing + projection, and writes outputs. |

**Data flow:** `main.py` → `Preprocessor.process()` → `Projector.setup_geometry()` → `Projector.project()` → save NIfTI / PNG / metadata.

---

## Requirements

- Python 3.11+
- A **CUDA-capable NVIDIA GPU** (required by TIGRE)
- Python packages:
  - `numpy`
  - `torch`
  - `monai`
  - `tigre`
  - `SimpleITK`
  - `imageio`

Install with:

```bash
pip install numpy torch monai SimpleITK imageio
# TIGRE has its own build/install steps (CUDA required):
# https://github.com/CERN/TIGRE
```

---

## Input data

- Input CTs must be in **NIfTI** format (`.nii` or `.nii.gz`).
- Place all input volumes in a single folder (see `input_dir` below).
- Each file is processed independently; outputs are written to a per-case subfolder named after the file stem.

---

## Configuration — what you need to set

All settings are in **`config.py`**. At minimum, point the pipeline at your data.

### 1. Paths (required)

```python
input_dir: str  = "../data/CTs"                              # <-- folder containing your CT NIfTI files
output_dir: str = f"../dataset_{n_angles}views_{end_angle}deg"  # <-- where results are written
```

### 2. Preprocessing parameters

```python
target_iso_mm: float = 1.875          # target isotropic voxel size (mm)
hu_window: Tuple = (-500, 1300)       # (H_min, H_max); also used as the fill/pad value (H_min)
orientation: str = "RAS"              # target anatomical orientation
```

### 3. Projection geometry

```python
sid: float = 1500.0     # source-to-detector distance (mm)
sod: float = 1000.0     # source-to-isocenter distance (mm); magnification M = SID / SOD
n_angles: int = 3       # number of projection angles
end_angle: float = 90.0 # angles are sampled uniformly in [0, end_angle]
mode: str = "parallel"  # "parallel" or "cone"
```

### 4. Detector

```python
det_pixels: Tuple = (512, 512)  # detector resolution (fixed pixel count)
det_padding: float = 0.6        # padding factor for the detector field of view
```
The physical detector size is derived automatically from the volume extent (height from the axial/cranio-caudal extent, width from the in-plane diagonal), scaled by `M` and `det_padding`. The **pixel pitch is computed automatically** as physical detector size ÷ pixel count — you do not set it directly.

### 5. Output options

```python
write_png: bool = True  # also export a normalized PNG per DRR alongside the NIfTI
```

## Usage

Once `config.py` points at your data:

```bash
python main.py
```

The pipeline will iterate over every `.nii`/`.nii.gz` file in `input_dir`, print progress per case, skip and report any file that fails, and write results into `output_dir`.

---

## Output

For each input CT, a subfolder `output_dir/<case_id>/` is created containing:

- `ct.nii.gz` — the preprocessed CT volume
- `<case_id>_drr_angle<AAA>.nii.gz` — one DRR per projection angle (line-integral projection)
- `<case_id>_drr_angle<AAA>.png` — optional normalized preview (if `write_png = True`)
- `metadata.json` — CT shape/spacing and the full DRR geometry (SID, SOD, detector size/spacing/pixels, angles)

> **Note on intensities:** DRRs are stored as accumulated attenuation line integrals. PNG previews are min–max normalized per image and are for visualization only — absolute intensity/scale is not comparable across images.

---

## Optional: Coregistration

`coreg.py` aligns anatomically homogeneous volumes (e.g., exclusively lumbar CTs) to a fixed reference using a single 3D 6-DOF rigid transform, initialized by a coarse rotation search and refined with multi-resolution mutual information. It is **not wired into `main.py`** and must be invoked separately.

To use it, add a reference path to your `Config`:

```python
# in config.py
reference_path: str = "../data/CTs/reference.nii.gz"   # <-- reference volume for registration
```
Voxels falling outside the moving image are filled with `hu_window[0]`. Optional registration hyperparameters (histogram bins, sampling percentage, learning rate, iterations, shrink factors, smoothing sigmas, coarse-search step, seed) can also be added to `Config`; sensible defaults are used otherwise.

---
## License

`<!-- Add license (e.g., MIT) here -->`
