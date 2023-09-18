import pytorch_lightning as pl
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics as tm
import sys 
sys.path.append('./modules/') 
import torch.optim as optim
from torch.nn.functional import mse_loss

def select_encoder(encoder_name, embedding_size):

    if encoder_name == "efficientnet": 
        encoder = timm.create_model('efficientnet_b0', pretrained=True, num_classes=embedding_size, in_chans=3) 

    elif encoder_name == "mobilenetv3": 
        encoder = timm.create_model('tf_mobilenetv3_large_075', pretrained=True, num_classes=embedding_size, in_chans=3) 

    return encoder

class Encoder(nn.Module):
    def __init__(self, encoder_name, embedding_size):
        super().__init__()

        self._encoder = select_encoder(encoder_name, embedding_size) 

    def forward(self, img):
        b, t ,w ,h, c = img.size()
        img = img.reshape(b*t,c,w,h)#.permute(0,2,3,1)
        embedding_obs = self._encoder(img).reshape(b,t,-1)

        return embedding_obs 

class Vpr(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self._model_lr = self.hparams.lr
        self._embedding_size = self.hparams.embedding_size
        self._encoder_name = self.hparams.encoder_name
        self._save_dir = self.hparams.save_dir
        self._batch_size = self.hparams.batch_size
        self._margin = self.hparams.margin
        self._pile_num_img = self.hparams.pile_num_img

        self._encoder = Encoder(self._encoder_name, self._embedding_size)
        self._gru = nn.GRU(self._embedding_size, 64, 2, batch_first=True)
        # self._gru2 = nn.GRU(self._embedding_size, 64, 2, batch_first=True)
        # self._lstm = nn.LSTM(self._embedding_size, 64, 2, batch_first=True)
        # self._lstm2 = nn.LSTM(self._embedding_size, 64, 2, batch_first=True)

        self._out_list = []
        self._img_list = []
        self._output_img_list = []

        self._tanh = nn.Tanh()

        self._fc = nn.Sequential(
                nn.Linear(64*2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
        )

    def forward(self, img):
        x = self._encoder(img)
        _, _, _ = x.size()

        # x, (h_n,c_n) = (self._lstm(x))
        x, _ = (self._gru(x))

        return x[:,-1,:]

    def calc_loss(self, batch, batch_size):
        anchor_img, positive_img, negative_img = batch
        features = [self(img, batch_size) for img in (anchor_img, positive_img, negative_img)]
        loss = F.triplet_margin_with_distance_loss(
                *features,
                distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                margin=self._margin)
        del batch
        return loss, features

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self._model_lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, features = self.calc_loss(batch, self._batch_size)
        self.log("train/loss", loss, prog_bar=False, logger=True, on_epoch=True)
        self.log("train/positive_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[1]).mean(),
            prog_bar=False, logger=True)
        self.log("train/negative_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[2]).mean(),
            prog_bar=False, logger=True)
        del batch
        return loss

    def validation_step(self, batch, batch_idx):
        loss, features = self.calc_loss(batch, self._batch_size)
        self.log("valid/loss", loss, prog_bar=False, logger=True)
        self.log("valid/positive_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[1]).mean(),
            prog_bar=False, logger=True)
        self.log("valid/negative_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[2]).mean(),
            prog_bar=False, logger=True)
        del batch
        return loss

    def test_step(self, batch, batch_idx):
        loss, features = self.calc_loss(batch, 1)
        anchor_img, positive_img, negative_img = batch
        del batch

        self.log("test/loss", loss, prog_bar=False, logger=True)
        self.log("test/loss", loss, prog_bar=False, logger=True)
        self.log("test/positive_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[1]).mean(),
            prog_bar=False, logger=True)
        self.log("valid/negative_distance",
            tm.functional.pairwise_euclidean_distance(features[0], features[2]).mean(),
            prog_bar=False, logger=True)

        from visualize import visualize_img
        output_img = visualize_img(anchor_img.squeeze(), positive_img.squeeze(), negative_img.squeeze())
        self._output_img_list.append(output_img)

        cost_ap = 1 - F.cosine_similarity(features[0], features[1]).mean().item()
        cost_an = 1 - F.cosine_similarity(features[0], features[2]).mean().item()

        # self._out_list.append(
        #         (
        #         features[0].mean().to('cpu').detach().numpy().copy(),
        #         features[1].mean().to('cpu').detach().numpy().copy(),
        #         features[2].mean().to('cpu').detach().numpy().copy()
        #         )

        # )
        self._out_list.append(
                (
                    cost_ap,
                    cost_an
                )

        )
    def on_test_epoch_end(self,):
        from visualize import save_figure 
        
        save_figure(self._out_list, self._output_img_list, self._save_dir)
 

