import torchvision.datasets as dset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from ipdb import set_trace as bp
import torch.utils.data as data
from PIL import Image
import os.path
import torch.nn.functional as F

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # if len(target) == 1:
        #     tt = np.zeros(91)
        #     tt[target[0]['category_id']] = 1
        #     target = tt
        # else:
        #     tt = np.zeros(91)
        #     target = tt
        fff1 = [t['category_id'] for t in target]
        tt = F.one_hot((torch.tensor(fff1)).to(torch.int64), num_classes=91)
        tt = (torch.sum(tt,axis=0) > 0).float()
        #tt = np.zeros(91)
        #for t in target:
        #   tt[t['category_id']] = 1
        #bp()
        
        target = tt
        
        #[1]:
        #    onehot[th['category_id']] = 1
        #label = onehot
        #trainset[step] = (trainset[step][0],onehot)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




def loadCocoData(numImagesPerClass, batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None):
    resNetOrInception = 0
    if resNetOrInception == 0:
        imgSize = 256
    else:
        imgSize = 299

    datapath = "/scratch/groups/rubin/alimirz1/"
    path2data=datapath + "train2017"
    path2val=datapath + "val2017"
    path2json=datapath + "annotations/instances_train2017.json"
    path2jsonval = datapath + "annotations/instances_val2017.json"

    transformations = transforms.Compose([
        #transforms.Resize((299,299)),
        #transforms.RandomRotation(10),
        #transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomVerticalFlip(0.5),
	transforms.RandomResizedCrop(imgSize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transformations_valid = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    coco_train = CocoDetection(root = path2data,
                                annFile = path2json,
                                transform = transformations)

    coco_valid = CocoDetection(root = path2val,
                                annFile = path2jsonval ,
                                transform = transformations_valid)

    validToInclude = []
    for jj in range(0,500):
        try:
            label = coco_train[jj][1]
            validToInclude.append(jj)
        except:
            print('pass')

    dataset_valid_new = torch.utils.data.Subset(coco_train, validToInclude)

    trainToInclude = []
    for jj in range(500,500 + numImagesPerClass):
        try:
            label = coco_train[jj][1]
            trainToInclude.append(jj)
        except:
            print('pass')

    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/1000train.out', np.array(trainToInclude ), delimiter=',')
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/1000val.out', np.array(validToInclude ), delimiter=',')

    sup_train = torch.utils.data.Subset(coco_train, trainToInclude)
    
    unsup_train = sup_train
    


    # If not None, unsupDatasetSize selects subset of the entire usable unsupervised dataset
    if unsupDatasetSize is not None:
        unsup_train = torch.utils.data.Subset(unsup_train, list(range(unsupDatasetSize)))

    
    dataset_test = coco_valid

    train_loader = DataLoader(
        sup_train, batch_size=batch_size, num_workers=4, shuffle=True)
    unsup_loader = DataLoader(
        unsup_train, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(
        dataset_valid_new, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        dataset_test, batch_size=1, num_workers=4)

    return train_loader, unsup_loader, valid_loader, test_loader


def balancedMiniDataset(trainset, size, limit, fullyBalanced=True):
    counter = np.zeros(91)
    iterating = True
    step = 500
    subsetToInclude = []
    subsetToNotInclude = []
    #subsetToNotInclude += list(range(step))
    while iterating and step < limit:
        #bp()
        try:
            label = np.array(trainset[step][1])
            cc = counter * label + label
            #if not np.all(cc[np.nonzero(cc)] > size): #np.all(counter + label <= size):
            if np.sum(label) == 1:
                counter += label
                print(counter, step)
                subsetToInclude.append(step)
            else:
                subsetToNotInclude.append(step)
            if np.min(counter) >= size:
                print("Completely Balanced Dataset")
                iterating = False
        except:
            print(step)
        if step%1000 == 0:
            print(step)
        step += 1
    subsetToNotInclude += list(range(step, len(trainset)))
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/2imgclassloosebalanced.out', np.array(subsetToInclude), delimiter=',')
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude) 
