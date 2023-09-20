import math
import torch
import torch.nn as nn
import numpy as np
#from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        #PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR +=peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    from skimage.measure import compare_ssim
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    # print("img.size",Img.shape)
    # SSIM = 0
    # for i in range(Img.shape[0]):
    #     SSIM = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], multichannel=False)
    # return (SSIM/Img.shape[0])
    SSIM = compare_ssim(Iclean[0, 0, :, :], Img[0, 0, :, :])
    return (SSIM )
def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


# # from models.Nets import MLP, CNNMnist, CNNCifar
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset

# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         #print(image)
#         return image, label



# ################################### data setup ########################################
# def data_setup(args):
    
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#     if args.dataset == 'mnist':
#         path = './data/mnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#                 dict_users = mnist_iid(dataset_train, args.num_users)
#                 # print(dict_users)
#                 # print(type(dict_users))
                
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             dict_users = mnist_noniid(dataset_train, args.num_users)
#     elif args.dataset == 'svhn':
#         path = './data/svhn'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
#         dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
#         dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
#         dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
#         #dataset_train = torch.utils.data.ConcatDataset([dataset_train])
#         dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#             # if 1:
#                 dict_users = svhn_iid(dataset_train, args.num_users)
#                     # print(dict_users)
#                     # print(type(dict_users))
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             exit('Error: only consider IID setting in SVHN')
#     elif args.dataset == 'emnist':
#         path = './data/emnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         # train_loader = datasets.CelebA('../data/', split='train', target_type='identity', download=True)
        
#         trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
#         dataset_train = datasets.EMNIST(path, split='balanced', train=True, download=True, transform=trans_emnist)
#         dataset_test = datasets.EMNIST(path, split='balanced', train=False, download=True, transform=trans_emnist)
#         args.num_classes = 10
#         if args.iid:
#             dict_users = emnist_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')
#     elif args.dataset == 'fmnist':
#         path = './data/fmnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
#         dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#                 dict_users = fmnist_iid(dataset_train, args.num_users)
#                 # print(dict_users)
#                 # print(type(dict_users))
                
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             exit('Error: only consider IID setting in fmnist')
#     elif args.dataset == 'cifar':
#         path = './data/cifar'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, 
#                                         transform=transforms.Compose([transforms.RandomHorizontalFlip(), 
#                                                                         transforms.RandomCrop(32, 4),
#                                                                         transforms.ToTensor(),
#                                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                         std=[0.229, 0.224, 0.225])]))
#         dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, 
#                                         transform=transforms.Compose([transforms.ToTensor(),
#                                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                             std=[0.229, 0.224, 0.225])]))
#         args.num_classes = 10
#         if args.iid:
#             dict_users = cifar_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')

#     ###############
#     ## adding support for vanilla svhn ###
#     # ####################
#     elif args.dataset == 'vanillasvhn':
#         path = './data/vanillasvhn'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])

#         dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
#         dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
#         dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
#         dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f:
#                     dict_users = dill.load(f)
#                     # print(dict_users)
#             else:
#             # if 1:
#                 dict_users = svhn_iid(dataset_train, args.num_users)
#                     # print(dict_users)
#                     # print(type(dict_users))
#                 with open(path  + "/dict_users.pik", 'wb') as f:
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f)
#         else:
#             exit('Error: only consider IID setting in SVHN')
#     ###############################  VANILLA SVHN  END ########################################

#     else:
#         exit('Error: unrecognized dataset')
    
#     return args, dataset_train, dataset_test, dict_users


# ###################### utils #################################################
# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
    
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

# def svhn_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
    
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         # all_idxs=random.shuffle(all_idxs)
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
    
#     return dict_users

# def fmnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from fashion MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     # print(dataset[0][0].size)
#     return dict_users

# def emnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from eMNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
  
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users



# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
 
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

# def emnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
   
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """

#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
