#!/usr/bin/env python
# coding: utf-8

import os
import json
import SimpleITK as sitk
import xml.etree.ElementTree as ET

from results_utils import (
    calculate_lesion_stats,
    dice_similarity_coefficient
)

def results_patient(refined_lesions_image, patient_num):
    """
    Processes the results of lesion segmentation by calculating lesion statistics and Dice Similarity Coefficient (DSC).

    Parameters:
    - atlas_path (str): Path to the atlas image (.nii.gz).
    - xml_path (str): Path to the XML file containing structure mappings.
    - refined_lesions_image_path (str): Path to the refined lesions image (.nii.gz).
    - annotated_image_path (str): Path to the annotated image for DSC calculation (.nii.gz).
    - patient_num (int): Patient number for file and folder references.

    Returns:
    - lesion_stats (list of dict): A list of dictionaries containing lesion statistics.
    - dsc (float): The Dice Similarity Coefficient between refined lesions and annotated lesions.
    """
    
    base_folder = f"../data/Patient-{patient_num}/"
    results_folder = os.path.join(base_folder, "results/")
    # Create directory if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Parse XML to create gray level to region name mapping
    xml_path = '../atlases/JulichBrainAtlas_3.0_areas_MPM_b_N10_nlin2ICBM152asym2009c.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    structures = root.find('Structures')

    gray_to_name = {}
    if structures is not None:
        for structure in structures.findall('Structure'):
            left_gray_value = int(structure.get('leftgrayvalue'))
            right_gray_value = int(structure.get('rightgrayvalue'))
            region_name = structure.text.strip()

            gray_to_name[left_gray_value] = region_name
            gray_to_name[right_gray_value] = region_name

    # Calculate lesion statistics with regional labels from atlas
    atlas = sitk.ReadImage('../atlases/JulichBrainAtlas_3.0_areas_MPM_b_N10_nlin2ICBM152asym2009c.nii.gz')
    lesion_stats = calculate_lesion_stats(refined_lesions_image, atlas)

    # Save lesion statistics to JSON
    json_stats_path = os.path.join(results_folder, "lesion_stats.json")
    with open(json_stats_path, 'w') as json_file:
        json.dump(lesion_stats, json_file, indent=4)

    # Print lesion statistics
    for lesion in lesion_stats:
        print(f"Label: {lesion['label']}")
        print(f"Region Name: {lesion['region_name']}")
        print(f"Volume (mm³): {lesion['volume_mm3']}")
        print(f"Largest Diagonal (mm): {lesion['largest_diagonal_mm']}")
        print(f"Lengths Along Axes (mm): {lesion['lengths_along_axes_mm']}")
        print()  # Blank line

    # Load the annotated image
    anotated_image = sitk.ReadImage(f'../data/Patient-{patient_num}/{patient_num}_ann2_isovox.nii.gz')

    # Calculate Dice Similarity Coefficient (DSC)
    dsc = dice_similarity_coefficient(refined_lesions_image, annotated_image)
    print(f"Dice Similarity Coefficient (DSC): {dsc}")
    
    # Save dsc to JSON
    json_dsc_path = os.path.join(results_folder, "dsc.json")
    with open(json_dsc_path, 'w') as json_file:
        json.dump(dsc, json_file, indent=4)

    return lesion_stats, dsc

