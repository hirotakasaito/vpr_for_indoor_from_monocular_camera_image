import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from modules.dataset import Dataset
from modules.data_module import DataModule
from modules.model import Vpr 

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--dataset-dir", type=str, default="/share/private/27th/hirotaka_saito/dataset/sq2/d_kan1/vpr/")
    parser.add_argument("-td", "--test-dataset-dir", type=str, default="/share/private/27th/hirotaka_saito/dataset/sq2/d_kan1/vpr_test/")
    parser.add_argument("-p", "--pretrained-checkpoint_dir", type=str, default=None)
    parser.add_argument("-es", "--embedding-size", type=int, default=256)
    # parser.add_argument("-en", "--encoder-name", type=str, default="mobilenetv3")
    parser.add_argument("-en", "--encoder-name", type=str, default="efficientnet")
    parser.add_argument("-sv", "--save-dir", type=str, default="./test_figures")
    parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    parser.add_argument("-e", "--num-epochs", type=int, default=10)
    parser.add_argument("-i", "--checkpoint_dir", type=str, default="./checkpoint_dir")
    parser.add_argument("-o", "--log-dir", type=str, default="./log")
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()

    dirs_name = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(args.log_dir, dirs_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, dirs_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    devices = torch.cuda.device_count()
    with open(os.path.join(log_dir, "args.txt"), mode="w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}, {value}\n")

    if args.pretrained_checkpoint_dir is None:
        model = Vpr(**vars(args))
    else:
        model = Vpr.load_from_checkpoint(args.pretrained_checkpoint_dir, **vars(args))

    model_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="valid/loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-loss={valid/loss:.6f}",
        auto_insert_metric_name=False
    )

    datamodule = DataModule(
            Dataset,
            args.dataset_dir,
            args.test_dataset_dir,
            args.seed,
            args.batch_size,
            args.num_workers,
    )

    trainer = pl.Trainer(
            callbacks = [model_checkpoint],
            logger = [TensorBoardLogger(log_dir)],
            max_epochs = args.num_epochs,
            accelerator="gpu",
            precision=16,
            benchmark=True,
            devices = devices,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()


