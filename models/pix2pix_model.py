from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.vgg import Vgg16
from MS_SSIM import ssim

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.use_gan = opt.use_GAN
        self.w_vgg = opt.w_vgg
        self.w_tv = opt.w_tv
        self.w_gan = opt.w_gan
        self.w_ss = opt.w_ss
        self.use_condition = opt.use_condition
        self.optimizers = []  # Define optimizers list

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.use_condition == 1:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            if opt.which_model_netD == 'multi':
                self.criterionGAN = networks.GANLoss_multi(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)

            self.criterionL1 = torch.nn.L1Loss()

            if torch.cuda.is_available():  # Check if CUDA is available
                self.vgg = Vgg16().to(self.device)  # Move Vgg16 model to GPU
            else:
                self.vgg = Vgg16().to(torch.device('cpu'))  # Move Vgg16 model to CPU

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        if self.use_condition == 1:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.use_condition == 1:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        if self.use_gan == 1:
            if self.use_condition == 1:
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            else:
                fake_AB = self.fake_B
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        self.real_B_features = self.vgg(self.real_B)
        self.fake_B_features = self.vgg(self.fake_B)
        self.loss_vgg = self.criterionL1(self.fake_B_features[1], self.real_B_features[1]) * 1 + \
                        self.criterionL1(self.fake_B_features[2], self.real_B_features[2]) * 1 + \
                        self.criterionL1(self.fake_B_features[3], self.real_B_features[3]) * 1 + \
                        self.criterionL1(self.fake_B_features[0], self.real_B_features[0]) * 1

        diff_i = torch.sum(torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :-1, :]))
        self.tv_loss = (diff_i + diff_j) / (320 * 256)

        X = ((self.real_B + 1) / 2)
        Y = ((self.fake_B + 1) / 2)

        self.loss_ssim = 1 - ssim(X, Y, data_range=1, size_average=True)

        self.loss_G = self.loss_G_GAN * self.w_gan + self.loss_G_L1 + \
                      self.loss_vgg * self.w_vgg + self.tv_loss * self.w_tv + \
                      self.w_ss * self.loss_ssim

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        
        if self.use_gan == 1:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
