import pytorch_lightning as pl
import timm
import torch
from torch import nn
import sys
sys.path.append('./modules/')
import torch.optim as optim
from torch.nn.functional import mse_loss

def select_encoder(encoder_name, embedding_size):

    if encoder_name == "efficientnet": 
        encoder = timm.create_model('efficientnet_b0', pretrained=True, num_classes=embedding_size, in_chans=3) 

    elif encoder_name == "mobilenetv3": 
        encoder = timm.create_model('tf_mobilenetv3_large_075', pretrained=True, num_classes=embedding_size, in_chans=3) 

    elif encoder_name == "swin_transformer": 
        encoder = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=embedding_size, in_chans=3) 

    return encoder


class Encoder(nn.Module):
    def __init__(self, encoder_name, embedding_size):
        super().__init__()

        self._encoder1 = select_encoder(encoder_name, embedding_size) 
        self._encoder2 = select_encoder(encoder_name, embedding_size) 

    def forward(self, input1, input2):
        b, t ,w ,h, c = input1.size()
        input1 = input1.reshape(b*t,c,w,h)#.permute(0,2,3,1)
        input2 = input2.reshape(b*t,c,w,h)#.permute(0,2,3,1)
        embedding_obs1 = self._encoder1(input1).reshape(b,t,-1)
        embedding_obs2 = self._encoder2(input2).reshape(b,t,-1)

        return embedding_obs1, embedding_obs2 

class Vpr(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self._model_lr = self.hparams.lr
        self._embedding_size = self.hparams.embedding_size
        self._encoder_name = self.hparams.encoder_name
        self._save_dir = self.hparams.save_dir
        self._batch_size = self.hparams.batch_size

        self._encoder = Encoder(self._encoder_name, self._embedding_size)
        # self._gru1 = nn.GRU(self._embedding_size, 64, 2, batch_first=True)
        # self._gru2 = nn.GRU(self._embedding_size, 64, 2, batch_first=True)
        self._lstm1 = nn.LSTM(self._embedding_size, 64, 2, batch_first=True)
        self._lstm2 = nn.LSTM(self._embedding_size, 64, 2, batch_first=True)

        self._out_list = []
        self._img_list = []
        self._output_img_list = []

        self._fc = nn.Sequential(
                nn.Linear(64*2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
        )

    def forward(self,img1, img2, batch_size):
        h0 = torch.zeros(2, batch_size, 64).to(img1.device)
        c0 = torch.zeros(2, batch_size, 64).to(img1.device)
        x1, x2 = self._encoder(img1, img2)
        # x1, _ = self._gru1(x1, h0)
        # x2, _ = self._gru2(x2, h0)
        x1, (_,_) = self._lstm1(x1, (h0,c0))
        x2, (_,_) = self._lstm2(x2, (h0,c0))

        train_output = self._fc(torch.cat([x1[:,-1,:], x2[:,-1,:]], dim=-1))

        return train_output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self._model_lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        img1, img2, gt = batch
        train_output = self(img1, img2, self._batch_size).squeeze()
        loss = mse_loss(train_output, gt)
        self.log("train/loss", loss, prog_bar=False, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, gt = batch
        valid_output = self(img1, img2, self._batch_size).squeeze()
        loss = mse_loss(valid_output, gt)
        self.log("valid/loss", loss, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        img1, img2, gt = batch
        test_output = self(img1, img2, 1).squeeze()
        loss = mse_loss(test_output, gt)

        self.log("test/loss", loss, prog_bar=False, logger=True)

        from visualize import visualize_img
        output_img = visualize_img(img1.squeeze(), img2.squeeze())
        self._output_img_list.append(output_img)

        self._out_list.append(
                (
                test_output.to('cpu').squeeze().detach().numpy().copy(),
                gt.squeeze().to('cpu').detach().numpy().copy()
                )

        )

        # self._img_list.append(
        #         (
        #             img1.squeeze().to('cpu').detach().numpy().copy(),
        #             img2.squeeze().to('cpu').detach().numpy().copy()
        #         )
        # )

    def on_test_epoch_end(self,):
        from visualize import save_figure 
        
        save_figure(self._out_list, self._output_img_list, self._save_dir)
 

