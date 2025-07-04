
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["t2.nii.gz"]
seg_filename = "seg.nii.gz"

base_dir = "/workspace/data/FedBCa_clean/"
image_dir = "all_centers"

def process_train():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [0.75, 0.75, 3.5]   # FedBCa median
    output_dir = "/workspace/data/FedBCa_clean/all_centers_processed"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1],
    )

def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":

    plan()
    process_train()

