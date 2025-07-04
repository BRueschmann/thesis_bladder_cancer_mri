import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from monai.utils import set_determinism
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last

set_determinism(123)

# -----------------------------------------------------------------------------
# Argument parser — every previously hard‑coded constant can now be set from SLURM
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train SegMamba on FedBCa bladder MRI")

parser.add_argument("--data_dir", type=str, default="./data/fullres/train",
                    help="Path to the prepared BraTS‑style training directory")
parser.add_argument("--logdir", type=str, default="./logs/segmamba",
                    help="Where checkpoints and TensorBoard logs will be written")
parser.add_argument("--max_epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--val_every", type=int, default=2,
                    help="Validate every N epochs")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--roi", nargs=3, type=int, default=[128, 128, 128],
                    metavar=("X", "Y", "Z"), help="3D patch size for training")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Trainer definition (unchanged except for arg wiring)
# -----------------------------------------------------------------------------
class BladderTrainer(Trainer):
    def __init__(self):
        super().__init__(env_type="pytorch",
                         max_epochs=args.max_epoch,
                         batch_size=args.batch_size,
                         device=args.device,
                         val_every=args.val_every,
                         num_gpus=torch.cuda.device_count() or 1,
                         logdir=args.logdir,
                         master_port=17759,
                         training_script=__file__)

        self.window_infer = SlidingWindowInferer(roi_size=args.roi,
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        from model_segmamba.segmamba import SegMamba
        self.model = SegMamba(in_chans=1,
                              out_chans=2,
                              depths=[2, 2, 2, 2],
                              feat_size=[48, 96, 192, 384])

        # ------------- balanced loss
        w_bg, w_fg = 0.1, 0.9
        class_w    = torch.tensor([w_bg, w_fg], device=args.device)
        self.ce   = nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        self.dice = DiceLoss(to_onehot_y=True, softmax=True,
                             include_background=False, reduction="mean")

        # ------------- optimiser
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.best_val_dice  = 0.0

    # ------------------------------------------------ get_input
    def get_input(self, batch):
        img   = batch["data"]           # B×1×D×H×W
        label = batch["seg"][:, 0].long()
        return img, label

    # ------------------------------------------------ train‑step
    def training_step(self, batch):
        img, seg = self.get_input(batch)
        logits   = self.model(img)
        loss     = 0.5 * self.ce(logits, seg) + 0.5 * self.dice(logits, seg)
        self.log("training_loss", loss, step=self.global_step)
        return loss

    # ------------------------------------------------ val‑step
    def validation_step(self, batch):
        img, seg = self.get_input(batch)
        pred     = self.model(img).softmax(1).argmax(1)
        inter    = ((pred == 1) & (seg == 1)).sum()
        denom    = (pred == 1).sum() + (seg == 1).sum() + 1e-6
        return (2.0 * inter.float() / denom).item()

    def validation_end(self, val_outputs):
        mean_dice = float(np.mean(val_outputs))
        self.log("val_dice", mean_dice, step=self.epoch)
        print(f"[epoch {self.epoch}]  val Dice = {mean_dice:.4f}")

        if mean_dice > self.best_val_dice:
            self.best_val_dice = mean_dice
            save_new_model_and_delete_last(
                self.model,
                os.path.join(args.logdir, "model",
                             f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    trainer = BladderTrainer()

    train_ds, val_ds, _ = get_train_val_test_loader_from_train(args.data_dir)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
