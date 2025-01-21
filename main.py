#!/usr/bin/env python
# coding: utf-8

import sys

def main():
    """
    Main entry point that executes all the steps:
      1) Preprocessing
      2) Coregistration
      3) Atlas processing
      4) Segmentation
      5) Results calculation

    Usage:
      python script_name.py [patient_num]

    If patient_num is not provided, it defaults to 1.
    """
    
    # If you want to pass the patient number via command line:
    if len(sys.argv) > 1:
        patient_num = int(sys.argv[1])
    else:
        patient_num = 1  # Default patient number if not specified

    print(f"Starting pipeline for patient {patient_num}...")

    # 1) Preprocessing
    print("\n--- Step 1: Preprocessing ---")
    preprocessing_patient(patient_num)

    # 2) Coregistration
    print("\n--- Step 2: Coregistration ---")
    corregistration_patient(patient_num)

    # 3) Atlas
    print("\n--- Step 3: Atlas ---")
    atlas_patient(patient_num)

    # 4) Segmentation
    print("\n--- Step 4: Segmentation ---")
    refined_lesions_image, results = segmentation_patient(patient_num)

    # 5) Results
    print("\n--- Step 5: Results ---")
    lesion_stats, dsc = results_patient(refined_lesions_image, patient_num)

    print("\nPipeline completed successfully!")
    print(f"Lesion stats and DSC saved for patient {patient_num}.")

if __name__ == "__main__":
    main()

