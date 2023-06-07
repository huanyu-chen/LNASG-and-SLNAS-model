import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10 , CIFAR100
from torchvision import transforms
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
import os
import random
import PIL
from PIL import Image

from tqdm import tqdm

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img
def get_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    if os.path.basename(args.data) == 'cifar10':
        train_dataset = CIFAR10(
            root=args.data,
            train=True,
            download=True,
            transform=train_transform,
        )
    else:
        train_dataset = CIFAR100(
            root=args.data,
            train=True,
            download=True,
            transform=train_transform,
        )

    indices = list(range(len(train_dataset)))
    train_size = int(0.8 * len(indices))
    train_indce = random.sample(indices,train_size)
    test_indce = list(set(indices) - set(train_indce))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_indce),
        pin_memory=True,
        num_workers=0,
    )

    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_indce),
        pin_memory=True,
        num_workers=0,
    )

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    if os.path.basename(args.data) == 'cifar10':
        valid_dataset = CIFAR10(
            root=args.data,
            train=False,
            download=False,
            transform=valid_transform,
        )
    else:
        valid_dataset = CIFAR100(
            root=args.data,
            train=False,
            download=False,
            transform=valid_transform,
        )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    #repeat_train_loader = RepeatedDataLoader(train_loader)
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader

import random
def get_retrain_loaders(args):
    train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomRotation((45,-45)),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data,'train'),
        transform=train_transform,
    )

    indices = list(range(len(train_dataset)))
    train_size = int(0.8 * len(indices))
    test_size = len(indices) - train_size
    train_indce = random.sample(indices,train_size)
    test_indce = list(set(indices) - set(train_indce))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_indce),
        #shuffle = True,
        pin_memory=True,
        num_workers=0,drop_last=True
    )
    reward_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_indce),
        pin_memory=True,
        num_workers=0,drop_last=True
    )
    valid_transform =transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
    valid_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data,'val'),
        transform=valid_transform)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,drop_last=True
    )
    #repeat_train_loader = RepeatedDataLoader(train_loader)
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader

############################

class RepeatedDataLoader():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = self.data_loader.__iter__()

    def __len__(self):
        return len(self.data_loader)

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()
        return batch
import numpy as np
class RandomChoice(transforms.RandomChoice):
    """Apply single transformation randomly picked from a list
    """
    def __init__(self, transforms,weights):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        self.weights = weights
    def __call__(self, img):
        #t = random.choices(self.transforms,weights = self.weights)[0]
        t = self.transforms[np.random.choice(len(self.transforms), 1, p=self.weights)[0]]
        return t(img)
transforms.RandomChoice = RandomChoice
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles,weights):
        self.angles = angles
        self.weights = weights
    def __call__(self, x):
        #angle = random.choices(self.angles,weights = self.weights)[0]
        angle = self.angle[np.random.choice(len(self.angle), 1, p=self.weights)[0]]
        return transforms.functional.rotate(x, angle,resample = PIL.Image.ANTIALIAS)
class Ram_ImageFolder(Dataset):
    def __init__(self, image_folder_path,transform):
        self.image_folder = torchvision.datasets.ImageFolder(image_folder_path, transform= transform)
        self.transforms = self.image_folder.transform
        print(len(self.image_folder.imgs))
        self.preloader_data = [np.array(Image.open(i[0]).convert('RGB')) for i in tqdm(self.image_folder.imgs)]
        self.preloader_label = torch.Tensor([i[1] for i in self.image_folder.imgs]).type(torch.long)
    def __len__(self):
        return len(self.image_folder.imgs) # of how many examples(images?) you have
    def __getitem__(self, index):
        img = Image.fromarray(self.preloader_data[index])
        return self.transforms(img), self.preloader_label[index]
def validation_ram(dataset,batch_size=140,sampler = None , shuffle=False, num_workers=0,pin_memory=True,drop_last=True):
    A = [i[0] for i in tqdm(torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=False, num_workers=0,pin_memory=False))]
    x = torch.cat(A)
    y = torch.Tensor(dataset.targets).type(torch.long)
    dataset =  torch.utils.data.TensorDataset(x,y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,pin_memory=pin_memory,drop_last=drop_last,sampler=sampler)


def AID_retrain_loader(args):
    data_transforms = {'train':transforms.Compose([
    #MyRotationTransform(angles=[0, 15, 30 , 45],weights=[0.55,0.15,0.15,0.15]),
    RandomChoice([transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
                  transforms.Resize((128,128),interpolation =PIL.Image.ANTIALIAS),
                  transforms.Resize((64,64),interpolation =PIL.Image.ANTIALIAS),
                  transforms.Resize((32,32),interpolation =PIL.Image.ANTIALIAS)],weights=[0.55,0.15,0.15,0.15]),
    transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
    #transforms.RandomRotation((45,-45)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224,interpolation =PIL.Image.ANTIALIAS),
    #transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]),

    'val': transforms.Compose([
        transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
        transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    }

    image_datasets ={x:torchvision.datasets.ImageFolder(os.path.join('{}'.format(args.data), x), #torchvision.datasets.CIFAR10(root='./data', train=(x == 'train'), download=False,#
                  transform=data_transforms[x])  for x in ['train', 'val']}
    dataloaders = {'train': torch.utils.data.DataLoader(Ram_ImageFolder(os.path.join('{}'.format(args.data), 'train'),data_transforms['train']), batch_size=args.batch_size,#16
                                             shuffle=True, num_workers=0,pin_memory=True,drop_last=True),
                'val':torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size,#16
                                             shuffle=False, num_workers=0,pin_memory=True,drop_last=False)}

    print('load training data to ram')
    train_loader =dataloaders['train']
    print('load validation data to ram')
    valid_loader = validation_ram(image_datasets['val'],shuffle=False, num_workers=0,pin_memory=True,drop_last=False)

    return train_loader, valid_loader
def get_retrain_loaders(args):
    train_transform = transforms.Compose([
                        RandomChoice([transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
                                      transforms.Resize((128,128),interpolation =PIL.Image.ANTIALIAS),
                                      transforms.Resize((64,64),interpolation =PIL.Image.ANTIALIAS),
                                      transforms.Resize((32,32),interpolation =PIL.Image.ANTIALIAS)],weights=[0.55,0.15,0.15,0.15]),
                        transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        transforms.RandomGrayscale(0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(224,interpolation =PIL.Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

    valid_transform = transforms.Compose([
                        transforms.Resize((256,256),interpolation =PIL.Image.ANTIALIAS),
                        transforms.CenterCrop((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    print('load training data to ram')
    train_dataset = Ram_ImageFolder(os.path.join(args.data, 'train'),train_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, 'val'), transform=valid_transform)

    indices = list(range(len(train_dataset)))
    train_size = int(0.8 * len(indices))
    test_size = len(indices) - train_size
    train_indce = random.sample(indices,train_size)
    reward_indce = list(set(indices) - set(train_indce))


    train_loader = torch.utils.data.DataLoader(train_dataset,sampler=SubsetRandomSampler(train_indce),
                                                    batch_size=args.batch_size, num_workers=0,pin_memory=True,drop_last=True)
    reward_loader = torch.utils.data.DataLoader(train_dataset,sampler=SubsetRandomSampler(reward_indce),
                                                    batch_size=args.batch_size*2, num_workers=0,pin_memory=True,drop_last=True)

    print('load validation data to ram')
    valid_loader = validation_ram(val_dataset,shuffle=True, num_workers=0,pin_memory=True,drop_last=False)


    #repeat_train_loader = RepeatedDataLoader(train_loader)
    repeat_reward_loader = RepeatedDataLoader(reward_loader)
    repeat_valid_loader = RepeatedDataLoader(valid_loader)

    return train_loader, repeat_reward_loader, repeat_valid_loader


def get_cifar_retrain_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
         #transforms.RandomGrayscale(0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),Cutout(16)#[transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ])
    train_dataset = CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform,
    )

    indices = list(range(len(train_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
        pin_memory=True,
        num_workers=0,drop_last=True
    )

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])
    valid_dataset = CIFAR10(
        root=args.data,
        train=False,
        download=False,
        transform=valid_transform,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    return train_loader, valid_loader
