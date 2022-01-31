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
import json

keepClasses = [25,5,85,24,16,7,22,70,21,20,23,19,13,86,10,9,11,15,17,88]

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

    def __init__(self, root, annFile, transform=None, target_transform=None, dropClasses=[]):
        from pycocotools.coco import COCO
        self.root = root
        #bp()
        self.coco = COCO(annFile[:-17] + ".json")
        f = open(annFile,"r")
        annotations = json.loads(f.read())
        self.dictImgID = {}
        for item in annotations["annotations"]:
            if str(item["image_id"]) in self.dictImgID:
                print("REPEAT")
                #self.dictImgID[str(item["image_id"])].append(item)
            else:
                self.dictImgID[str(item["image_id"])] = item

        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.dropClasses = dropClasses

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        #bp()
        coco = self.coco
        img_id = self.ids[index]
        #print(img_id)
        target = self.dictImgID[str(img_id)]["category_id"]
        #target = target[keepClasses]

        #ann_ids = coco.getAnnIds(imgIds=img_id)
        #target = coco.loadAnns(ann_ids)
        #fff1 = [t['category_id'] for t in target]
        #tt = F.one_hot(torch.tensor(fff1), num_classes=91)
        #tt = (torch.sum(tt,axis=0) > 0).float()        
        #target = tt
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if np.sum(np.array(target)[self.dropClasses]) > 0:
            raise Exception("class too rare! discarding")
        else:
            return img, torch.tensor(target, dtype=torch.float32)


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
    if fullyBalanced:
        dropClasses = [50,39,80,74,43,37,31,84,48,89,40,27,45,12,26,29,30,66,68,69,71,83,0]
    else:
        dropClasses = [45,12,26,29,30,66,68,69,71,83,0]
    resNetOrInception = 0
    if resNetOrInception == 0:
        imgSize = 256
    else:
        imgSize = 299

    datapath = "/scratch/groups/rubin/alimirz1/"
    path2data=datapath + "train2017"
    path2val=datapath + "val2017"
    path2json=datapath + "annotations/instances_train2017_clean_top20.json"
    path2jsonval = datapath + "annotations/instances_val2017_clean_top20.json"

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

    #validToInclude = []

    #sup_train, dataset_valid_new = balancedMiniDataset(coco_train, numImagesPerClass, len(coco_train))
    supToInclude = np.loadtxt("/home/users/alimirz1/SemisupervisedAttention/saved_batches/coco_splits/"+str(numImagesPerClass)+"_per_top20class.csv").astype(int)
    validToInclude = np.loadtxt("/home/users/alimirz1/SemisupervisedAttention/saved_batches/coco_splits/430_per_top20class_validation.csv").astype(int)[:2000]
    

    #for jj in range(0,numValidation ):
    #    try:
    #        label = coco_train[jj][1]
    #        validToInclude.append(jj)
    #    except:
    #        print('pass')
    sup_train = torch.utils.data.Subset(coco_train, supToInclude)
    dataset_valid_new = torch.utils.data.Subset(coco_train, validToInclude)
    #bp()
    
    #for jj in range(numValidation ,numValidation + numImagesPerClass):
    #allInstancesCounter = np.zeros(91)
    #singleInstancesCounter = np.zeros(91)
    #numErrors = 0
    #for jj in range(len(coco_train)):
    #    if jj % 500 == 0:
    #        print(jj, "out of", len(coco_train))
    #        print("Num Errors: ", numErrors)
    #    try:
    #        label = coco_train[jj][1]
    #        boolList = torch.gt(label,0).numpy().tolist()
    #        allInstancesCounter[boolList]+=1
    #        if sum(boolList) == 1:
    #             singleInstancesCounter[boolList]+=1
    #        trainToInclude.append(jj)
    #    except:
    #        numErrors += 1
    #bp()
    
    #ranked = np.argsort(singleInstancesCounter)
    #largest_indices = ranked[::-1]
    #largest_values = singleInstancesCounter[largest_indices]
    #print(list(zip(largest_indices, largest_values)))
    #bp()
    
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/1000train.out', np.array(trainToInclude ), delimiter=',')
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/1000val.out', np.array(validToInclude ), delimiter=',')

    #sup_train = torch.utils.data.Subset(coco_train, trainToInclude)
    if useNewUnsupervised:
        unsup_train = dataset_valid_new
    else:
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
    counter = np.zeros(20)
    #counter += size
    
    #if fullyBalanced:
    #    dropClasses = [50,39,80,74,43,37,31,84,48,89,40,27,45,12,26,29,30,66,68,69,71,83,0]
    #else:
    #    dropClasses = [45,12,26,29,30,66,68,69,71,83,0]
    #for ccc in dropClasses:
    #    counter[dropClasses] = size
    #for ccc in keepClasses:
    #    counter[keepClasses] = 0
    iterating = True
    step = 0
    subsetToInclude = []
    subsetToNotInclude = []
    #subsetToNotInclude += list(range(step))
    while iterating and step < limit:
        #bp()
        try:
            #bp()
            label = np.array(trainset[step][1])
            #bp()
            if np.all(counter + label <= size) and (not fullyBalanced or np.sum(label).item() == 1):
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

    #subsetToNotInclude += list(range(step, len(trainset)))
    #subsetToNotInclude = subsetToNotInclude[:10000]
    #while len(subsetToNotInclude) < 10000:
    #    try:
    #        label = np.array(trainset[step][1])
    #        subsetToNotInclude.append(step)
    #    except:
    #        print(step)
    #subsetToNotInclude += list(range(step, len(trainset)))
    np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/coco_splits/'+str(size)+'_per_top20class.csv', np.array(subsetToInclude), delimiter=',')

    np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/coco_splits/'+str(size)+'_per_top20class_validation.csv', np.array(subsetToNotInclude), delimiter=',')
    bp()
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude) 
