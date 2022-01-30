import torch.optim as optim
import os; os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
import cv2
import sys
import json
import distutils.util
sys.path.append("./")

from data_loader.new_pascal_runner import loadPascalData
from metrics.SupervisedMetrics import Evaluator

import tqdm
import argparse

device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')



def get_resnet_base():
    resnet = models.resnet50(pretrained=True)
    last_block = nn.Sequential(*list(resnet.get_submodule('layer4.2').children())[:-1], nn.Tanh())
    model = nn.Sequential(*list(resnet.children())[:-1], last_block).to(device) # drop softmax and average pool 2D
    # replace last ReLU activation with Tanh activation for embedding

    return model


# source: https://github.com/leokarlin/LaSO/blob/master/oneshot/setops_models/setops.py
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

# source: https://github.com/leokarlin/LaSO/blob/master/oneshot/setops_models/setops.py
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


def setup_model(num_classes=20):
    base_model = get_resnet_base().to(device)
    setops_model = SetOpsModule().to(device)
    classifier_model = ClassifierModule(num_classes=num_classes).to(device)

    laso_classifier = LaSOClassifier(base_model, classifier_model)

    # classification_loss = nn.MultiLabelSoftMarginLoss() # paper uses BCE, code uses Soft Margin?
    classification_loss = nn.BCEWithLogitsLoss()
    recon_loss = nn.MSELoss()

    return laso_classifier, base_model, setops_model, classifier_model, classification_loss, recon_loss


def train(epochs=300, batch_size=16, images_per_class=2, eval_freq=10, data_dir='../data', download_data=False):
    suptrainloader, _, validloader, testloader = loadPascalData(images_per_class, data_dir=data_dir, download_data=download_data, batch_size=batch_size//2, useNewUnsupervised=False)

    laso_classifier, base_model, setops_model, classifier_model, classification_loss, reconstruction_loss = setup_model()
    optimizer = optim.Adam(list(base_model.parameters()) + list(setops_model.parameters()) + list(classifier_model.parameters()), lr=0.0001, betas=(0.9,0.999)) #end 0.002? Decay by 30 percent every time validation loss plateaus I think


    base_model.train() # If we set this to .eval(), it is equivalently to freezing our resnet base model

    setops_model.train()
    classifier_model.train()

    train_loader = iter(suptrainloader)

    # create evaluator
    evaluator = Evaluator()

    def load(train_loader=train_loader):
        try:
            return next(train_loader)
        except StopIteration:
            train_loader = iter(suptrainloader)
            return None, None

    for i in tqdm.tqdm(range(epochs)):
        # source: https://github.com/leokarlin/LaSO/blob/master/scripts_coco/train_setops_stripped.py
        # get batch of data
        while True:
            train_features_a, train_labels_a = load(train_loader)
            if train_features_a is None:
                break
            train_features_b, train_labels_b = load(train_loader)
            if train_features_b is None:
                break

            train_features_a = train_features_a.to(device)
            train_features_b = train_features_b.to(device)

            train_labels_a = train_labels_a.to(device).float()
            train_labels_b = train_labels_b.to(device).float()

            target_a = train_labels_a.byte()
            target_b = train_labels_b.byte()

            # reset optimizer
            optimizer.zero_grad()

            embed_a = base_model(train_features_a.to(device))
            embed_b = base_model(train_features_b.to(device))
            # print("embed_a", embed_a.grad)

            embed_a = embed_a.view(embed_a.size(0), -1)
            embed_b = embed_a.view(embed_b.size(0), -1)
            # print("Embed shapes", embed_a.shape, embed_b.shape) # torch.Size([8, 2048, 1, 1]) torch.Size([8, 2048, 1, 1]

            output_a = classifier_model(embed_a)
            output_b = classifier_model(embed_b)

            outputs_setopt = setops_model(embed_a, embed_b)

            fake_a, fake_b, a_S_b, b_S_a, a_U_b, b_U_a, a_I_b, b_I_a, \
            a_S_b_b, b_S_a_a, a_I_b_b, b_I_a_a, a_U_b_b, b_U_a_a, \
            a_S_b_I_a, b_S_a_I_b, a_S_a_I_b, b_S_b_I_a = \
                        [classifier_model(o) for o in outputs_setopt]
            fake_a_em, fake_b_em, a_S_b_em, b_S_a_em, a_U_b_em, b_U_a_em, a_I_b_em, b_I_a_em, \
            a_S_b_b_em, b_S_a_a_em, a_I_b_b_em, b_I_a_a_em, a_U_b_b_em, b_U_a_a_em, \
            a_S_b_I_a_em, b_S_a_I_b_em, a_S_a_I_b_em, b_S_b_I_a_em = outputs_setopt

            loss_class = classification_loss(output_a, train_labels_a) + classification_loss(output_b, train_labels_b)
            loss_class_laso_out = classification_loss(fake_a, train_labels_a) + classification_loss(fake_b, train_labels_b)
            return_loss_class = loss_class.clone().detach().item()

            # loss for reproducing embed_a and embed_b through setops_model
            loss_recon = reconstruction_loss(embed_a, fake_a_em) + reconstruction_loss(embed_b, fake_b_em)
            return_loss_recon = loss_recon.clone().detach().item()

            #
            # Calculate the target setopt operations
            #

            target_a_I_b = target_a & target_b
            target_a_U_b = target_a | target_b
            target_a_S_b = target_a & ~target_a_I_b
            target_b_S_a = target_b & ~target_a_I_b

            target_a_I_b = target_a_I_b.float()
            target_a_U_b = target_a_U_b.float()
            target_a_S_b = target_a_S_b.float()
            target_b_S_a = target_b_S_a.float()

            loss_class_S = classification_loss(a_S_b, target_a_S_b) + classification_loss(b_S_a, target_b_S_a)
            loss_class_U = classification_loss(a_U_b, target_a_U_b)
            loss_class_I = classification_loss(a_I_b, target_a_I_b)

            loss_class_S += classification_loss(a_S_b_b, target_a_S_b) + classification_loss(b_S_a_a, target_b_S_a)
            loss_class_S += classification_loss(a_S_a_I_b, target_a_S_b) + classification_loss(b_S_a_I_b, target_b_S_a) +\
                            classification_loss(b_S_b_I_a, target_b_S_a) + classification_loss(a_S_b_I_a, target_a_S_b)
            loss_class_U += classification_loss(a_U_b_b, target_a_U_b) + classification_loss(b_U_a_a, target_a_U_b)
            loss_class_I += classification_loss(a_I_b_b, target_a_I_b) + classification_loss(b_I_a_a, target_a_I_b)

            loss_recon_S = reconstruction_loss(a_S_b_em, a_S_b_b_em) + reconstruction_loss(a_S_b_em, a_S_a_I_b_em) + \
                            reconstruction_loss(a_S_b_em, a_S_b_I_a_em)
            loss_recon_S += reconstruction_loss(b_S_a_em, b_S_a_a_em) + reconstruction_loss(b_S_a_em, b_S_a_I_b_em) + \
                            reconstruction_loss(b_S_a_em, b_S_b_I_a_em)
            return_recon_S = loss_recon_S.item()

            loss_class_U += classification_loss(b_U_a, target_a_U_b)
            loss_class_I += classification_loss(b_I_a, target_a_I_b)

            loss_recon_U = reconstruction_loss(a_U_b_em, b_U_a_em)
            loss_recon_I = reconstruction_loss(a_I_b_em, b_I_a_em)
            return_recon_U = loss_recon_U.clone().detach().item()
            return_recon_I = loss_recon_I.clone().detach().item()
            loss = loss_class

            loss += loss_class_laso_out
            loss += loss_recon
            loss += loss_class_S
            loss += loss_recon_S
            loss += loss_class_U
            loss += loss_recon_U
            loss += loss_class_I
            loss += loss_recon_I

            loss.backward()
            optimizer.step()

        if not (i + 1) % eval_freq:
            # evaluate
            print(f"Loss: {loss.item():3.3f} | Classification loss: {return_loss_class:3.3f} | Reconstruction loss: {loss_recon.item():3.3f}")

            base_model.eval()
            setops_model.eval()
            classifier_model.eval()

            optimizer.zero_grad()

            evaluator.evaluateModelSupervisedPerformance(laso_classifier, validloader, classification_loss, device, optimizer, storeLoss=True)
            evaluator.plotLosses()

            base_model.train()
            setops_model.train()
            classifier_model.train()

    return base_model, setops_model, classifier_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="number of epochs to run", type=int, default=100)
    parser.add_argument("--images_per_class", "-i", help="number of images per class to train on", type=int, default=2)
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--eval_freq", help="frequency of epochs to run evaluation method", type=int, default=4)
    parser.add_argument("--data_dir", help="where Pascal VOC is stored", type=str, default="../data")
    parser.add_argument("--download_data", help="flag to download data or not", action="store_true")

    args = parser.parse_args()

    laso_models = train(epochs=args.epochs,
                        images_per_class=args.images_per_class,
                        eval_freq=args.eval_freq,
                        data_dir=args.data_dir,
                        download_data=args.download_data,
                        batch_size=args.batch_size)