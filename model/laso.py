import os
import numpy as np

import torchvision.models as models
import torch
from torch import nn

import sys
sys.path.append("./")


device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')

def get_resnet_base():
    resnet = models.resnet50(pretrained=True)
    last_block = nn.Sequential(*list(resnet.get_submodule('layer4.2').children())[:-1], nn.Tanh())
    model = nn.Sequential(*list(resnet.children())[:-2], last_block).to(device) # drop softmax and average pool 2D
    # replace last ReLU activation with Tanh activation for embedding

    return model


# NOTE: Try different values of dropout

class LaSOModule(nn.Module):
    def __init__(self, inner_dim, latent_dim=2048): # or latent dim 1024?
        super(LaSOModule, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(latent_dim*2, inner_dim))

        self.linear_block = nn.Sequential(
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(inner_dim, inner_dim),
            nn.BatchNorm1d(inner_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(inner_dim, latent_dim),
            nn.Tanh()
        )

        self.float()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        ab = torch.cat((a, b), dim=1)

        out = self.l1(ab)
        genFeatureVec = self.linear_block(out)
        return genFeatureVec.float()

class SetOpsModule(nn.Module):
    def __init__(
            self,
            input_dim: int=2048,
            S_latent_dim: int=2048,
            I_latent_dim: int=2048,
            U_latent_dim: int=2048,
            block_cls: nn.Module=LaSOModule,
            **kwds):

        super(SetOpsModule, self).__init__()

        self.subtract_op = block_cls(
            inner_dim=input_dim,
            latent_dim=S_latent_dim,
            **kwds
        )
        self.intersect_op = block_cls(
            inner_dim=input_dim,
            latent_dim=I_latent_dim,
            **kwds
        )
        self.union_op = block_cls(
            inner_dim=input_dim,
            latent_dim=U_latent_dim,
            **kwds
        )

        self.float()

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)

        a_S_b = self.subtract_op(a, b)
        b_S_a = self.subtract_op(b, a)

        a_S_b_b = self.subtract_op(a_S_b, b)
        b_S_a_a = self.subtract_op(b_S_a, a)

        a_I_b = self.intersect_op(a, b)
        b_I_a = self.intersect_op(b, a)

        a_S_b_I_a = self.subtract_op(a, b_I_a)
        b_S_a_I_b = self.subtract_op(b, a_I_b)
        a_S_a_I_b = self.subtract_op(a, a_I_b)
        b_S_b_I_a = self.subtract_op(b, b_I_a)

        a_I_b_b = self.intersect_op(a_I_b, b)
        b_I_a_a = self.intersect_op(b_I_a, a)

        a_U_b = self.union_op(a, b)
        b_U_a = self.union_op(b, a)

        a_U_b_b = self.union_op(a_U_b, b)
        b_U_a_a = self.union_op(b_U_a, a)

        out_a = self.union_op(a_S_b_I_a, a_I_b)
        out_b = self.union_op(b_S_a_I_b, b_I_a)

        return out_a, out_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
               a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
               a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a


class ClassifierModule(nn.Module):

    def __init__(self, input_dim=2048, num_classes=1000, **kwargs):
        super(ClassifierModule, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.fc(x)

class LaSOClassifier(nn.Module):

    def __init__(self, base_model, classifier):
        super(LaSOClassifier,self).__init__()
        self.base_model = base_model
        self.classifier = classifier

    def forward(self,x:torch.Tensor):
        embed = self.base_model(x)
        embed = embed.view(embed.size(0),-1)

        return self.classifier(embed)



class LaSO(nn.Module):

    def __init__(self, device=device, num_classes=20):
        super(LaSO,self).__init__()
        self.base_model = get_resnet_base().to(device)
        self.setops_model = SetOpsModule().to(device)
        self.classifier_model = ClassifierModule(num_classes=num_classes).to(device)

    def forward(self, x:torch.Tensor):
        # split input into 2
        a = x[:x.size(0)//2]
        b = x[x.size(0)//2:]

        embed_a = self.base_model(a)
        embed_b = self.base_model(b)

        # reshape
        embed_a = embed_a.view(embed_a.size(0), -1)
        embed_b = embed_a.view(embed_b.size(0), -1)

        # classifier output (regular multi-label scores)
        output_a = self.classifier_model(embed_a)
        output_b = self.classifier_model(embed_b)

        outputs_setopt = self.setops_model(embed_a, embed_b)

        # outputs of set operations from classifier
        outputs_classifier = [self.classifier_model(o) for o in outputs_setopt]

        return embed_a, embed_b, output_a, output_b, outputs_setopt, outputs_classifier


class LaSOLoss(nn.Module):
    def __init__(self, classification_criterion, reconstruction_criterion):
        super(LaSOLoss, self).__init__()

        self.classification_criterion = classification_criterion
        self.reconstruction_criterion = reconstruction_criterion

    def forward(self, laso_outputs, target):
        # unpack laso outputs (values that LaSO(train_features) should return)
        embed_a, embed_b, output_a, output_b, outputs_setopt, outputs_classifier = laso_outputs

        # split target labels into 2
        target_a = target[:target.size(0)//2]
        target_b = target[target.size(0)//2:]

        # unpack outputs_setopt
        fake_a_em, fake_b_em, a_S_b_em, b_S_a_em, a_U_b_em, b_U_a_em, a_I_b_em, b_I_a_em, \
        a_S_b_b_em, b_S_a_a_em, a_I_b_b_em, b_I_a_a_em, a_U_b_b_em, b_U_a_a_em, \
        a_S_b_I_a_em, b_S_a_I_b_em, a_S_a_I_b_em, b_S_b_I_a_em = outputs_setopt

        # unpack outputs_classifier
        fake_a, fake_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
        a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
        a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a = outputs_classifier

        loss_class = self.classification_criterion(output_a, target_a.float()) + self.classification_criterion(output_b, target_b.float())
        loss_class_laso_out = self.classification_criterion(fake_a, target_a.float()) + self.classification_criterion(fake_b, target_b.float())

        # loss for reproducing embed_a and embed_b through setops_model
        loss_recon = self.reconstruction_criterion(embed_a, fake_a_em) + self.reconstruction_criterion(embed_b, fake_b_em)

        target_a = target_a.byte() # need bit operators
        target_b = target_b.byte() # need bit operators

        # find set operations target values
        target_a_I_b = target_a & target_b
        target_a_U_b = target_a | target_b
        target_a_S_b = target_a & ~target_a_I_b
        target_b_S_a = target_b & ~target_a_I_b

        target_a_I_b = target_a_I_b.float()
        target_a_U_b = target_a_U_b.float()
        target_a_S_b = target_a_S_b.float()
        target_b_S_a = target_b_S_a.float()

        # classification loss for set operations
        loss_class_S = self.classification_criterion(a_S_b, target_a_S_b) + self.classification_criterion(b_S_a, target_b_S_a)
        loss_class_U = self.classification_criterion(a_U_b, target_a_U_b)
        loss_class_I = self.classification_criterion(a_I_b, target_a_I_b)

        loss_class_S += self.classification_criterion(a_S_b_b, target_a_S_b) + self.classification_criterion(b_S_a_a, target_b_S_a)
        loss_class_S += self.classification_criterion(a_S_a_I_b, target_a_S_b) + self.classification_criterion(b_S_a_I_b, target_b_S_a) +\
                        self.classification_criterion(b_S_b_I_a, target_b_S_a) + self.classification_criterion(a_S_b_I_a, target_a_S_b)
        loss_class_U += self.classification_criterion(a_U_b_b, target_a_U_b) + self.classification_criterion(b_U_a_a, target_a_U_b)
        loss_class_I += self.classification_criterion(a_I_b_b, target_a_I_b) + self.classification_criterion(b_I_a_a, target_a_I_b)

        # LaSO reconstruction loss for embedding learning
        # helps with embedding mode collapse supposedly
        loss_recon_S = self.reconstruction_criterion(a_S_b_em, a_S_b_b_em) + self.reconstruction_criterion(a_S_b_em, a_S_a_I_b_em) + \
                        self.reconstruction_criterion(a_S_b_em, a_S_b_I_a_em)
        loss_recon_S += self.reconstruction_criterion(b_S_a_em, b_S_a_a_em) + self.reconstruction_criterion(b_S_a_em, b_S_a_I_b_em) + \
                        self.reconstruction_criterion(b_S_a_em, b_S_b_I_a_em)

        loss_class_U += self.classification_criterion(b_U_a, target_a_U_b)
        loss_class_I += self.classification_criterion(b_I_a, target_a_I_b)

        loss_recon_U = self.reconstruction_criterion(a_U_b_em, b_U_a_em)
        loss_recon_I = self.reconstruction_criterion(a_I_b_em, b_I_a_em)


        loss = loss_class

        loss += loss_class_laso_out
        loss += loss_recon
        # don't include subtraction for now (a little weird)
        # loss += loss_class_S
        # loss += loss_recon_S
        loss += loss_class_U
        loss += loss_recon_U
        loss += loss_class_I
        loss += loss_recon_I

        return loss


if __name__ == "__main__":
    # RUN FOR SANITY CHECK
    from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
    laso = LaSO()
    laso_loss = LaSOLoss(BCEWithLogitsLoss(), MSELoss())

    laso.zero_grad()
    # this should be None
    print("Classifier fully connected layer gradient", laso.classifier_model.fc.weight.grad)

    x = laso(torch.randn((8,3,244,244)).to(device))
    loss = laso_loss(x, torch.randint(0,2,(8,20)).to(device))

    loss.backward()
    # this should be nonzero
    print("Classifier fully connected layer gradient", laso.classifier_model.fc.weight.grad)