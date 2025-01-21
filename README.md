# msautomaticseg

**MRI Multiple Sclerosis Lesion Segmentation Pipeline**

## Table of Contents
1. [Overview](#overview)
2. [Data Organization](#data-organization)
3. [Pipeline Steps](#pipeline-steps)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Limitations](#limitations)

## Overview

This repository describes a pipeline designed for rapid automatic segmentation of lesions in brain MRI, their 3D visualization, and anatomical characterization (see the overview diagram).
The principal goal is to provide a unified framework that integrates:

- **Preprocessing** of MRI sequences (T1, T2, FLAIR).
- **Correction** of field inhomogeneities and other distortions.
- **Registration** of multiple sequences.
- **Probabilistic atlas registration** and generation of a **similarity map**.
- **Tissue segmentation** and **lesion refinement**.
- **Anatomical characterization** of lesions (volume, subcortical location, etc.).

Significant effort was placed on selecting the most effective methods at each stage based on comparative reviews and minimizing external dependencies to facilitate potential deployment in a clinical environment.


## Data organization

The pipeline works on a folder structure similar to:

```bash
data/
  Patient-1/
    1_T1_isovox.nii.gz
    1_T2_isovox.nii.gz
    1_FLAIR_isovox.nii.gz
    ...
    preprocessed/
    transforms/
    atlas/
    segmentation/
    results/
  Patient-2/
    ...
atlases/
  ICBM_Template.nii.gz
  ICBM_Template_mask.nii.gz
  atlas_csf.nii.gz
  atlas_gray.nii.gz
  atlas_white.nii.gz
  JulichBrainAtlas_3.0_areas_MPM_b_N10_nlin2ICBM152asym2009c.nii.gz
  JulichBrainAtlas_3.0_areas_MPM_b_N10_nlin2ICBM152asym2009c.xml
```

Patient-x/ subfolders (preprocessed/, transforms/, atlas/, segmentation/, results/) is created and populated as the pipeline advances.

MRI sequences (T1, T2, FLAIR) should be placed under the Patient-x folder. 
Atlas files should be stored separately under atlases/.

Example output files for each step are provided for Patient-1, considering T1, T2, FLAIR input sequences.

## Pipeline steps

### 1. Preprocessing

These initial steps prepare the MRI volumes for subsequent analysis.

**Skull Stripping**  
The skull, meninges, and other extracerebral tissues are removed from T1-weighted images to facilitate registration, bias correction, and intensity-based tissue classification. A Python-based version of the BET algorithm (FSL) is employed, provided by the `brainextractor` library. The outcome is a binary brain mask.
Output: brain_mask_t1.nii.gz saved at Patient-1/

**Noise Reduction (Anisotropic Diffusion)**  
After isolating the brain, anisotropic diffusion filtering is appllied to reduce noise while preserving important structural details and lesion edges. This step enhances the efficiency of later bias field correction and segmentation.
Output: T1_adf.nii.gz, t2_adf.nii.gz, flair_adf.nii.gz saved at data/Patient-1/preprocessing/

**Bias Field Correction**  
The N4 Bias Field Correction algorithm (available in ANTs) is used to address inhomogeneities in the MRI intensities caused by magnetic field or RF coil non-uniformities. As a result, the corrected images exhibit a more uniform intensity scale, facilitating tissue classification.
Output: t1_corrected.nii, t2_corrected.nii.gz, flair_corrected.nii.gz saved at data/Patient-1/preprocessing/

### 2. Coregistration

Coregistration is the process of aligning multiple imaging modalities to ensure that anatomical structures correspond accurately across all images. This is important for subsequent tissue segmentation steps.
FLAIR is chosen as the reference ('fixed') image because lesion segmentation ultimately focuses on the FLAIR sequence, and any spatial distortion in FLAIR could compromise the accuracy of lesion delineation. Produces warped images (e.g., t1_registered.nii.gz, t2_registered.nii.gz) and a transformed brain mask.

**Rigid Transformation**  
Estimates translations and rotations (6 parameters), serving as an initial global alignment that simplifies later optimization stages.
This rigid transformation is estimated between the FLAIR and each moving sequence (T1, T2, and the brain mask).

**Multiresolution Affine Transformation**  
Starting from the rigid result, an affine transform is estimated (rotation, translation, scaling, and shearing).  
Uses mutual information (Mattes metric) and a multiresolution scheme to progressively refine alignment from coarse to fine scales.
We use a multiresolution approach, initially aligning low-resolution versions of the images, then refining at higher resolutions.

**Resampling**  
The moving image is resampled using the final affine transform with *b-spline* interpolation to preserve image quality. Any negative intensities introduced by interpolation are thresholded.

Output: brain_mask_registered.nii.gz, t1_registered.nii.gz, t2_registered.nii.gz saved at data/Patient-1/transforms/
---

### 3. Atlas Registration and Similarity Map

This step generates the spatial priors required for initializing the EM algorithm.
These priors comprise the registered reference probabilistic atlases (GM, WM, CSF) and a similarity map, which is generated by comparing the patient’s T1 sequence with a reference atlas.

1. **Probabilistic Atlases**  
   - The ICBM probabilistic atlases provide probability maps of gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF), compiled from large-population MR images.
   - Since probabilistic atlases represent only the likelihood of each tissue type (without full anatomical intensities), they are typically aligned using the same transformations computed for a reference atlas. An additional affine registration refines the alignment of each probabilistic map to the target image (which is the coregistered T1 image, because reference atlas is T1-w).

2. **Similarity Map (NCC)**  
   - A reference atlas (ICBM Template T1) after the brain extraction step is registered to the patient’s T1 sequence.  
   - Both volumes undergo intensity normalization and histogram matching so that they share a comparable intensity distribution.  
   - The Normalized Cross-Correlation (NCC) is computed in small local windows (3×3×3) between the atlas and patient’s T1 sequence at each voxel.  
   - The result is a similarity map (0–1 range) that quantifies the local agreement between the atlas and patient’s anatomy. This map is used in the GMM initialization of the segmentation by locally weighting the atlas-based priors. This improves the characterization of tissue anatomy and the identification of atypical intensities, including potential lesions.

   Output: affine_transform.mat, atlas_gray.nii.gz, atlas_white.nii.gz, atlas_csf.nii.gz, ICBM_Template_moved.nii.gz, t1_atlas_similarity.nii.gz,  saved at data/Patient-1/atlas
---

### 4. Tissue Segmentation

Healthy brain tissues (GM, WM, CSF) are approximated using a Gaussian Mixture Model (GMM) in a multi-dimensional intensity space (T1, T2, and FLAIR).

We implement an **Expectation-Maximization (EM) algorithm**:
1. **Initialization**  
Means and covariances and weights for each tissue class are initialized using the probabilistic atlases combined with the similarity map.

2. **Expectation**  
Calculates posterior probabilities (“responsibilities”) that each voxel belongs to each class, given the current parameters.  
Atlas priors are integrated into the Gaussian probability density model.

3. **Maximization**  
Updates the model parameters (means, covariances, and mixing weights for each tissue class) based on posterior probabilities from the Expectation step.

4. **Convergence Criterion**  
The log-likelihood is evaluated. We measure changes in the 
Iteration continues until the log-likelihood change falls below (10^{-6}) or a maximum iteration limit (set at 200) is reached.

After convergence, we derive posterior probability maps for each tissue. A final classification volume is built by selecting the highest-probability class at each voxel. 

Output: None
---

### 5. Lesion Segmentation

Lesion segmentation involves accurately identifying hyperintense lesions in the FLAIR sequence. This step includes the following stages:

1. **Contrast Enhancement**  
To amplify the contrast between lesions and surrounding tissues, an exponential transformation and a contrast enhancement algorithm proposed by Souplet et al. are applied. This facilitates the identification and precise delineation of FLAIR hyperintense lesions.

2. **Threshold Estimation**  
The lesion-segmentation threshold for FLAIR images is typically derived from the gray matter (GM) intensity statistics. The standard procedure involves estimating the mean (μ) and standard deviation (σ) of GM (with σ often inferred from the full width at half maximum, FWHM), and then applying an empirical factor (α) according to the relationship T=μ+ασ. However, as an exponential transformation has been applied to the FLAIR image, the underlying GM intensity distribution has changed. The new threshold must therefore be determined from the parameters of the transformed GM distribution.

3. **Refinement and Component Analysis**  
False positive lesion voxels are excluded through the following steps:
   - Small connected components below a minimum size threshold are removed.  
   - Unrealistic lesions (such as lesions within ventricles or hyperintensities in cortical regions) are removed using anatomical and tissue priors.
   - Lesion voxels are verified to predominantly lie within WM tissue.  
   - Morphological operations are applied to ensure lesion components are contiguous and to fill small holes, finalizing the lesion mask.

Output: refined_lesions_image.nii.gz saved at data/Patient-1/segmentation/

---

### 6. Lesion Characterization

- Lesion statistics: for each lesion, calculates volume (mm³), largest diagonal length, and length along principal axis. 
- Anatomical region: 
The final lesion mask is overlaid onto the Julich Brain Atlas, which has been previously co-registered to the patient’s anatomical space. A dictionary mapping atlas region-coded intensity values to corresponding anatomical labels was also generated. For each lesion, the subcortical region with the highest overlap is identified. **See Note below.**
These measurements are exported (in CSV or JSON format) for subsequent analysis or clinical review.

In cases where the Patient-x files belong to a training or evaluation database containing expert-annotated delineations:
- Dice Similarity Coefficient (DSC): The DSC is used to compare predicted lesions with expert-annotated lesion masks, providing a quantitative measure of segmentation accuracy. The DSC results are saved in a JSON file.

Output: lesion_stats.json, dsc.json, lesions_3d_surface.vtk saved to data/Patient-1/results

**Note:**
Mapping white matter lesions with a subcortical anatomical atlas can provide some information:
- Helps identify whether a lesion is adjacent to or disrupting major subcortical nuclei (as the thalamus or basal ganglia) or key pathways such as the internal capsule or corticospinal tract and provides insight into potential functional impacts (e.g., motor deficits if the internal capsule is affected).
- As the atlas includes some white matter pathways, lesion mapping can help infer the disrupted connections.

However, these labels should not be considered as a reliable source of information. White matter lesions often affect complex networks of pathways that are not captured by atlases designed for gray matter localization. Functional information (as connectivity analysis using DTI/tractography) is required to fully understand the implications of white matter lesions, which anatomical atlases alone cannot provide.


## Requirements

- Python (>= 3.8)
- NiBabel: For reading/writing NIfTI images.
- ANTsPy: For advanced registration and image processing.
- SimpleITK: For additional registration, IO, and filtering functionality.
- NumPy and SciPy: Fundamental libraries for numeric computations.
- scikit-learn: If your expectation-maximization is dependent on it (optional).
- Additional libraries depending on your environment, such as glob, json, xml.etree.ElementTree for file management and results handling.

Install these via pip or conda:

```bash
pip install nibabel antspyx SimpleITK numpy scipy scikit-learn
```

## Usage

1. Clone or download this repository.
2. Install dependencies (see Requirements).
3. Run the pipeline by executing the main() function in your script:
```bash
python main.py [patient_number]
```
The pipeline will sequentially execute Preprocessing, Coregistration, Atlas Registration, Segmentation, and Results. All outputs will be placed in the respective subfolders within data/Patient-31/.


## Limitations

- Data quality: The pipeline relies on adequate SNR (Signal-to-Noise Ratio) and correct intensity calibration. Extremely noisy or poorly acquired scans may undermine results.
- Atlas dependency: Atlas-based tissue priors assume a “typical” brain morphology. Significant pathologies (large tumors, extreme atrophy) can cause mismatches.
- Registration accuracy: Coregistration errors can propagate through the pipeline, impacting segmentation performance.
- Medical use: This pipeline is for research only. It should not be used as a diagnostic tool.
