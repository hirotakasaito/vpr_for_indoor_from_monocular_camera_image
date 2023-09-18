from random import sample
import torch
import pytorch_lightning as pl
import argparse
import os

from modules.model import Vpr

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained-checkpoint_dir", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    os.makedirs(args.output_dir,exist_ok=True)
    model = Vpr.load_from_checkpoint(args.pretrained_checkpoint_dir).eval()
    sample_img = torch.randn(1,10,224,224,3)
    torch.jit.save(
        model.to_torchscript(
            file_path=None, method='trace', example_inputs=sample_img
        ),
        f=os.path.join(args.output_dir,"vpr_output.pt")
    )

def test():
    args = arg_parse()
    model = torch.jit.load(os.path.join(args.output_dir,"vpr_output.pt"))
    sample_img = torch.randn(1,10,224,224,3)
    model_output = model(sample_img)
    print("model shape")
    print(model_output.shape)

if __name__ == "__main__":
    main()
    #Make sure if saved torch script works properly
    test()
