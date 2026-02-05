# epi1_superres_miccai

MICCAI 2026 conference abstract project based on the epi1 framework.

## Project Overview

This project extends the EPI1 framework for a super-resolution application in diffusion MRI.

### Extended Encoding Scheme

Traditional diffusion MRI encoding considers:
- **B-values**: Diffusion weighting strength
- **B-vectors**: Gradient directions for diffusion sensitization

We extend this encoding by incorporating **multiple phase encoding directions** (PEDs).

### Phase Encoding for Super-Resolution

Conventionally, two opposite phase encoding directions (e.g., AP and PA) are acquired solely for **susceptibility distortion correction**. However, susceptibility-induced geometric distortions create:
- **Local compression** in some regions
- **Local stretching** in other regions

This compression/stretching effect encodes spatial information at different effective resolutions along the phase encoding axis. By acquiring multiple phase encoding directions (beyond just two opposite directions), we can:
1. Sample the same anatomy with different local magnification factors
2. Exploit these varying effective resolutions for **super-resolution reconstruction**
3. Combine distortion correction with resolution enhancement in a unified framework

### Project Goal: Protocol Optimization

**Optimization task:** Select a subset of volumes from the giant 8-PED dataset that maximizes super-resolution quality.

**Selection dimensions:**
- Phase encoding direction
- B-value
- B-vector

**Baselines for comparison:**
- **Single PED:** 1× typical DKI measurement
- **Reverse PED pair:** 2× typical DKI measurement (standard distortion correction approach)
- **Standard processing pipeline:** MRtrix and FSL (topup/eddy for distortion correction)

**Constraint:** Total number of volumes ≤ baseline (same measurement time)

**Goal:** Find the optimal volume selection that outperforms baselines in super-resolution reconstruction.

### Acquisition Protocol

Based on a **DKI (Diffusion Kurtosis Imaging)** protocol, which requires multiple b-values (typically b=0, 1000, 2000 s/mm²) and sufficient gradient directions for kurtosis tensor estimation.

### Available Data

**High-resolution reference dataset:**
- Full DKI acquisition across **8 phase encoding directions**
- 4 pairs of reverse phase encoding directions
- Each pair rotated by 45° steps (0°, 45°, 90°, 135°)
- This is 8× a typical DKI measurement, enabling extensive protocol optimization experiments

**Typical DKI measurement (baseline):**
- Single phase encoding direction (or one reverse pair)
- The optimization goal: achieve super-resolution gains while using only 1× typical measurement time

**Low-resolution dataset:**
- Initially: down-sampled from high-resolution data
- Later: potentially additional acquired data
- Used for protocol optimization experiments

**T1-weighted image:**
- Separate anatomical acquisition
- Reference for b=0 structural comparison

### Comparison Strategy

- **Diffusion parameters (FA, MK):** Compare low-res reconstructions against high-res reference dataset
- **b=0 images:** Compare against T1-weighted anatomical reference

### Reconstruction Method

Joint reconstruction across multiple phase encoding directions, integrating distortion correction and super-resolution in a unified framework. See the **epi1 repository** for implementation details.

### Evaluation

**Structural super-resolution assessment:**
- Compare against **T1-weighted images** as anatomical reference (gold standard for structural detail)
- Evaluate improved resolution in derived diffusion parameters:
  - **FA (Fractional Anisotropy)** - from DTI fit
  - **MK (Mean Kurtosis)** - from DKI fit

Compare against a standard DKI protocol (same measurement time, conventional dual PED approach) to demonstrate enhanced structural detail in FA and MK maps.

## External Dependencies

### epi1 Framework

epi1 is included as a **git submodule** at `./submodules/epi1`.

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>

# Or initialize after clone
git submodule update --init --recursive
```
