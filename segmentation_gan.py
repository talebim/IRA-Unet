import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from inc_res_att_unet import base_Unet
from Discriminator import Network
from loss import loss_CE, loss_Discriminator

pl.seed_everything(42)


class SimpleGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.G = base_Unet()
        self.D = Network()
        self.ls = loss_CE()
        self.adversarial_loss = loss_Discriminator()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, data):
        return torch.nn.functional.softmax(self.G(data))


    def training_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        mri = batch['image']
        mask = batch['mask']
        # mask = mask.float()
        pred = self(mri)

        X_real = torch.mul(mri, mask)
        X_fake = torch.mul(mri, pred)

        ##########################
        # Optimize Discriminator #
        ##########################
        real_o = self.D(X_real.float())
        real_o = F.sigmoid(real_o)

        tens_one = torch.ones(real_o.shape, device=torch.device('cuda'))
        tens_zero = torch.zeros(real_o.shape, device=torch.device('cuda'))

        errD_real = self.adversarial_loss(real_o, tens_one)

        fake_o = self.D(X_fake.detach())
        fake_o = F.sigmoid(fake_o)
        errD_fake = self.adversarial_loss(fake_o, tens_zero)

        errD = errD_real + errD_fake

        self.log("d_loss", errD)


        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################

        fake_o = self.D(X_fake.float())
        fake_o = F.sigmoid(fake_o)
        errGD = self.adversarial_loss(fake_o, tens_one)
        errGG = self.ls(pred, mask)
        errG = errGG + (0.1*errGD)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log("g_loss", errG)

        return {'g_loss': errG, "d_loss": errD}

    def validation_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        mri = batch['image']
        mask = batch['mask']
        pred = self(mri)

        _, pre = torch.max(pred, dim=1)
        pre_sq = torch.unsqueeze(pre, dim=1)

        _, msk = torch.max(mask, dim=1)
        msk_sq = torch.unsqueeze(msk, dim=1)

        X_real = torch.mul(mri, mask)
        X_fake = torch.mul(mri, pred)

        ##########################
        # Optimize Discriminator #
        ##########################
        real_o = self.D(X_real.float())
        real_o = F.sigmoid(real_o)

        tens_one = torch.ones(real_o.shape, device=torch.device('cuda'))
        tens_zero = torch.zeros(real_o.shape, device=torch.device('cuda'))

        errD_real = self.adversarial_loss(real_o, tens_one)

        fake_o = self.D(X_fake.detach())
        fake_o = F.sigmoid(fake_o)
        errD_fake = self.adversarial_loss(fake_o, tens_zero)

        errD_val = (errD_real) + (errD_fake)

        ######################
        # Optimize Generator #
        ######################

        fake_o = self.D(X_fake.float())
        fake_o = F.sigmoid(fake_o)
        errGD = self.adversarial_loss(fake_o, tens_one)
        errGG = self.ls(pred, mask)
        errG_val = errGG + (0.1* errGD)

        self.log_dict({"g_loss_val": errG_val, "d_loss_val": errD_val})

        return {'g_loss_val': errG_val, "d_loss_val": errD_val, 'G_out_val':pre_sq,'G_out_msk':msk_sq, 'D_out_val':fake_o}

    def training_epoch_end(self, outputs):
        avg_loss_G = torch.stack([x["g_loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("LossG",
                                          {'G_train':avg_loss_G},
                                          self.current_epoch)

        avg_loss_D = torch.stack([x["d_loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("LossD",
                                          {'D_train':avg_loss_D},
                                          self.current_epoch)


    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["g_loss_val"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("LossG",
                                          {'G_val':avg_loss},
                                          self.current_epoch)

        avg_loss_D = torch.stack([x["d_loss_val"] for x in outputs]).mean()
        self.logger.experiment.add_scalars("LossD",
                                          {'D_val':avg_loss_D},
                                          self.current_epoch)


        self.logger.experiment.add_image("generator_images", outputs[0]["G_out_val"],self.current_epoch, dataformats='NCHW')
        self.logger.experiment.add_image("mask_images", outputs[0]["G_out_msk"],self.current_epoch, dataformats='NCHW')


        return {'val_g_loss': avg_loss.item()}

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters())
        d_opt = torch.optim.Adam(self.D.parameters())
        return g_opt, d_opt
