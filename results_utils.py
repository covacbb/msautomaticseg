#!/usr/bin/env python
# coding: utf-8


import SimpleITK as sitk
import numpy as np
from scipy.ndimage import find_objects


def calculate_lesion_stats(refined_lesions_image, atlas):
    """
    Calculates the statistical information about lesions: volume, 
    region name, bounding box dimensions, and largest diagonal length.

    Parameters:
    - refined_lesions_image (SimpleITK.Image): A labeled image of refined lesions where each lesion has a unique label.
    - atlas (SimpleITK.Image): An atlas image providing anatomical labels for the brain regions.

    Returns:
    - lesion_stats (list of dict): A list of dictionaries, each containing the following keys:
    - "label" (int): The lesion label.
    - "region_name" (str): The most common region name corresponding to the lesion in the atlas.
    - "volume_mm3" (float): The lesion's volume in cubic millimeters.
    - "largest_diagonal_mm" (float): The largest diagonal of the lesion's bounding box in millimeters.
    - "lengths_along_axes_mm" (ndarray): The lengths of the lesion along each axis of its bounding box in millimeters.
    """
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(refined_lesions_image)
    cc_array = sitk.GetArrayFromImage(cc_img)
    atlas_array = sitk.GetArrayFromImage(atlas)
    
    lesion_stats = []
    for lesion_label in range(1, cc_filter.GetObjectCount() + 1):
        lesion_slice = cc_array == lesion_label
        lesion_voxels = np.sum(lesion_slice)
    
        if lesion_voxels > 3:
            # Calculate volume in mm³
            volume = lesion_voxels  # since 1 voxel = 1 mm³
            
            # Extract region indices
            lesion_indices = find_objects(lesion_slice)[0]  # Get bounding box slice for the lesion
            lesion_subarray = cc_array[lesion_indices]  # Extract lesion subarray
            atlas_subarray = atlas_array[lesion_indices]
    
            # Most common atlas label in the lesion
            region_labels, counts = np.unique(atlas_subarray[lesion_slice[lesion_indices]], return_counts=True)
            most_common_region = region_labels[np.argmax(counts)]
            region_name = gray_to_name.get(most_common_region, "Unknown Region")
    
            # Diagonal and lengths along axes
            bbox_lengths = np.array([s.stop - s.start for s in lesion_indices])
            largest_diagonal = np.linalg.norm(bbox_lengths)
            lengths_along_axes = bbox_lengths
            
            lesion_stats.append({
                "label": lesion_label,
                "region_name": region_name,
                "volume_mm3": volume,
                "largest_diagonal_mm": largest_diagonal,
                "lengths_along_axes_mm": lengths_along_axes
            })

            #with open(f'../data/Patient-{patient_num}/segmentation/lesion_stats.json', 'w') as json_file:
            #    json.dump(lesion_stats, json_file, indent=4)

    return lesion_stats


def dice_similarity_coefficient(sitk_mask1, sitk_mask2):
    """
    Calculate the Dice Similarity Coefficient between two binary masks.
    
    Parameters:
    - mask1 (np.array): a binary mask (e.g., numpy array) where lesions are marked
    - mask2 (np.array): another binary mask for comparison
    
    Returns:
    - float: the Dice similarity coefficient between the two masks
    """
    # Convert masks to boolean arrays (1 for lesion, 0 for non-lesion)
    mask1 = sitk.GetArrayFromImage(sitk_mask1)
    mask2 = sitk.GetArrayFromImage(sitk_mask2)
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    
    # Compute Dice coefficient
    if total == 0:  # both masks are empty
        return 1.0
    else:
        return 2 * intersection / total

