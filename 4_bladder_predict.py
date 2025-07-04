import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.prediction import Predictor
from light_training.trainer import Trainer

set_determinism(123)

# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SegMamba bladder prediction script")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the trained SegMamba checkpoint (\.pt file)")
parser.add_argument("--save_dir", type=str, default="./prediction_results/segmamba",
                    help="Directory where prediction NIfTI files will be written")
parser.add_argument("--data_dir", type=str, default="./data/fullres/train",
                    help="Directory containing the pre‑processed cases (same as during training)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Static hyper‑parameters (unchanged from original script)
# -----------------------------------------------------------------------------
env         = "pytorch"
max_epoch   = 1000
batch_size  = 2
val_every   = 2
num_gpus    = 1
device      = "cuda:0"
patch_size  = [128, 128, 128]

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, model_path, save_dir,
                 device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip="localhost", master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)

        self.model_path = model_path
        self.save_dir   = save_dir
        self.patch_size = patch_size
        self.augmentation = False

    # ───────────── label conversion: binary tumour / background
    def convert_labels(self, labels):
        return (labels == 1).float()        # B×1×D×H×W

    def get_input(self, batch):
        image      = batch["data"]
        label      = batch["seg"]
        properties = batch["properties"]
        label      = self.convert_labels(label)
        return image, label, properties

    def define_model_segmamba(self):
        from model_segmamba.segmamba import SegMamba
        model = SegMamba(in_chans=1,
                         out_chans=2,
                         depths=[2, 2, 2, 2],
                         feat_size=[48, 96, 192, 384])

        new_sd = self.filte_state_dict(torch.load(self.model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()

        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                            sw_batch_size=2,
                                            overlap=0.5,
                                            progress=True,
                                            mode="gaussian")
        predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

        os.makedirs(self.save_dir, exist_ok=True)
        return model, predictor, self.save_dir

    def validation_step(self, batch):
        image, label, properties = self.get_input(batch)
        model, predictor, save_path = self.define_model_segmamba()

        # prediction
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)
        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        model_output = model_output.argmax(dim=0, keepdim=True)  # B×1×...

        # Dice for quick sanity check
        output_np = model_output[0].cpu().numpy()
        label_np  = label[0, 0].cpu().numpy()
        d         = dice(output_np, label_np)
        print([d])

        # save NIfTI
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output,
                              raw_spacing=[1, 1, 1],
                              case_name=properties['name'][0],
                              save_dir=save_path)
        return 0

    # -------------------------------------------------------------------------
    # helper
    # -------------------------------------------------------------------------
    def filte_state_dict(self, sd):
        if "module" in sd:
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        return new_sd

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           model_path=args.model_path,
                           save_dir=args.save_dir,
                           device=device,
                           logdir="",
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17751,
                           training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(args.data_dir)
    trainer.validation_single_gpu(test_ds)
