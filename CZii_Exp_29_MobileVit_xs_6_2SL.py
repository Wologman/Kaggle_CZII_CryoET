# Note: This version was set up to ignore the 'empty' class that I made myself.
# Though it performed OK from my own CV calculations, it did not generalise
# to the leaderboard as well as the earlier models that included 'empty'

# #Standard Python
import os
import gc
import random
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

#Machine Learning
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score
from tqdm import tqdm

#PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from pytorch_lightning.loggers import CSVLogger

#Medical Imaging
from monai.transforms import (
    Lambda,
    Compose,
    OneOf,
    RandRotate90,
    RandFlip,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAdjustContrast,
)
from monai.utils import set_determinism

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


class TrainConfig:
    '''Wrapper class for training hyperparameters'''
    def __init__(self):
        self.EXPERIMENT_NAME = 'Exp_29'
        self.RUN_ID = 'Run_01'
        self.DESCRIPTION = ''
        self.EXTRA_CORES = 4 #None will result in the max cpu count being used.
        self.FOCAL_GAMMA = 2
        self.MODEL_NAME = 'mobilevit_xs.cvnets_in1k'
        #'mobilevitv2_175.cvnets_in22k_ft_in1k_384' 
        #mobilevit_xs.cvnets_in1k    
        #'mobilevit_s.cvnets_in1k'#efficientvit_b3.r288_in1k, 'resnet18.a1_in1k', 'eca_nfnet_l0.ra2_in1k'
        #Others less promising: 'tinynet_e.in1k', 'efficientnet_lite0.ra_in1k', 'resnet34.a1_in1k'
        #'test_resnet.r160_in1k', convnext_tiny.in12k_ft_in1k, efficientvit_b3.r288_in1k, deit_tiny_patch16_224
        self.FOCAL_ALPHA = False
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.5
        self.RANDOM_SEED = 2025
        self.MAX_EPOCHS = 40
        self.EPOCHS_BACKBONE_FROZEN_0 = 5 #5 Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS_0 = 4 # If unfreezing the backbone, ufreeze this many layers
        self.EPOCHS_BACKBONE_FROZEN_1 = 8 #8 # Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS_1 = 6 # If unfreezing the backbone, ufreeze this many layers
        self.LOSS_FUNCTION = 'BinaryFocalLoss' #'BinaryFocalLoss' #'BinaryFocalLoss' # Could also consider 'CrossEntropy' or 'FocalLoss'
        self.PATIENCE = 4 # Stop training if no improvement
        self.BATCH_SIZE = 16 #For eval only
        self.TRAIN_BATCH_SIZE = 64 #64
        self.LEARNING_RATE = .0002 #5e-2
        self.INITIAL_LR = 1e-5
        self.WARMUP_EPOCHS = 2
        self.LR_CYCLE_LENGTH = 35 #10
        self.MIN_LR = 1e-6
        self.LR_DECAY = 0.5 #.5
        self.WEIGHT_DECAY = 1e-2
        self.RANDOMISE_TOP = None
        self.NUM_RANDOMISED = 1


class PreTrainConfig(TrainConfig):
    def __init__(self):
        super().__init__()
        self.MAX_EPOCHS = 50
        self.FIXED_EPOCHS = 25
        self.EPOCHS_BACKBONE_FROZEN_0 = 6 # Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS_0 = 6# If unfreezing the backbone, ufreeze this many layers
        self.EPOCHS_BACKBONE_FROZEN_1 = 12 # Set to None to keep backbone Frozen.
        self.UNFREEZE_LAYERS_1 = 8 # If unfreezing the backbone, ufreeze this many layers
        self.LOSS_FUNCTION = 'CrossEntropy' #'BinaryFocalLoss' #'BinaryFocalLoss' # Could also consider 'CrossEntropy' or 'FocalLoss'
        self.PATIENCE = 6 # Stop training if no improvement
        self.LEARNING_RATE = .001 #5e-2
        self.INITIAL_LR = .0001
        self.WARMUP_EPOCHS = 4
        self.LR_CYCLE_LENGTH = 10 #10
        self.MIN_LR = 1e-6
        self.LR_DECAY = 0.2 #.5
        self.RANDOMISE_TOP = True
        self.NUM_RANDOMISED = 4
        


class ImageConfig:
    '''Wrapper class for image processing parameters'''
    INPUT_MEAN = [ 0.485, 0.456, 0.406 ] # values from ImageNet.
    INPUT_STD = [ 0.229, 0.224, 0.225 ] #  values from ImageNet.
    NUM_PATCHES = 4


class DataConfig:
    '''Wrapper class data related parameters'''

    #Weighting for the training loss function
    CLASS_WEIGHTS = {'apo-ferritin':1,
                    'beta-amylase':1,
                    'beta-galactosidase':3,
                    #'empty': 1,
                    'ribosome': 1,
                    'thyroglobulin' :2,
                    'virus-like-particle':1}

    #Thresholds for inference
    THRESHOLDS = {'apo-ferritin': 0.4,
                  'beta-amylase': 0.4,
                  'beta-galactosidase': 0.4,
                  #'empty': 0.4,
                  'ribosome': 0.4,
                  'thyroglobulin' : 0.4,
                  'virus-like-particle': 0.4}
    TEST_FRACTION = 0.1
    VAL_FRACTION = 0.1

    def __init__(self):
        self.CLASS_NAMES = list(self.CLASS_WEIGHTS.keys())


class EvalConfig:
    THRESHOLDS = {'apo-ferritin': 0.4,     
                  'beta-amylase': 0.4,
                  'beta-galactosidase': 0.4,
                  #'empty': 0.4,
                  'ribosome': 0.4,
                  'thyroglobulin' : 0.4,
                  'virus-like-particle': 0.4}

    WEIGHTS = {'apo-ferritin': 1,
               'beta-amylase': 0,
               'beta-galactosidase': 2,
               #'empty': 0,
               'ribosome': 1,
               'thyroglobulin' : 2,
               'virus-like-particle': 1}


class Paths:
    '''Wrapper class for filepaths'''
    DATA_FOLDER_NM = 'Data'
    EXPS_FOLDER_NM = 'Experiments'
    IMAGE_FOLDER_NM = '40x40_crops_uint8'
    SYN_FOLDER_NM = '40X40_synthetic_crops'
    WEIGHTS_FOLDER_SUFFIX = '_weights'
    METRICS_FN_SUFFIX = '_monitor_metrics.png'
    BEST_WEIGHTS_FN_SUFFIX = '_best_weights.pt'
    RESULTS_FOLDER_NM = 'Results'  # Increment or name this to name the results folder
    MODELS_FOLDER_NM = 'Models'

    def __init__(self, experiment_name, run_id=None, ):
        _project_dir = Path(__file__).resolve().parent.parent
        _experiment_dir = _project_dir / self.DATA_FOLDER_NM / self.EXPS_FOLDER_NM / experiment_name

        self.image_dir = _project_dir / self.DATA_FOLDER_NM  / self.IMAGE_FOLDER_NM
        self.syn_image_dir = _project_dir / self.DATA_FOLDER_NM  / self.SYN_FOLDER_NM
        self.results_dir = _experiment_dir / self.RESULTS_FOLDER_NM
        self.models_dir = _experiment_dir / self.MODELS_FOLDER_NM
        self.weights_pth = self.models_dir / f'{run_id}{self.WEIGHTS_FOLDER_SUFFIX}'
        self.final_weights_pth = self.weights_pth / f'{run_id}{self.BEST_WEIGHTS_FN_SUFFIX}'
        self.train_metrics_pth = self.results_dir / f'{run_id}{self.METRICS_FN_SUFFIX}'
        self.pretrain_metrics_pth = self.results_dir / f'{run_id}_pretrain{self.METRICS_FN_SUFFIX}'
        self.plots_dir = self.results_dir / f'{run_id}_plots'

        for fldr in [self.results_dir, self.models_dir, self.plots_dir, self.weights_pth]:
            if not os.path.exists(fldr):
                fldr.mkdir(parents=True, exist_ok=True) 


def set_hardware(cfg):
    if cfg.EXTRA_CORES is None:
        num_workers = cpu_count()-1
    else:
        num_workers = cfg.EXTRA_CORES
    gpu = torch.cuda.is_available()
    accelerator = 'gpu' if gpu else 'cpu'
    torch.set_float32_matmul_precision('medium') #could try setting to 'high' at expense of speed

    print(f'Loading data with {num_workers + 1} CPU cores')
    print(f"Using torch {torch.__version__} "
        f"{torch.cuda.get_device_properties(0) if accelerator == 'gpu' else 'CPU'}")

    if accelerator =='gpu':
        gc.collect()
        torch.cuda.empty_cache()

    return num_workers, accelerator


class Colour:
    '''Turn shell print statements bold blue'''
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_map_score(target_df, pred_df, average='macro'):
    col_sums = target_df.sum()
    mask = col_sums >= 1 #keeping this in to avoid division by 0
    targs_arr = target_df.loc[:,mask].copy().values
    preds_arr = pred_df.loc[:,mask].copy().values
    if average is None:
        scores_vals = average_precision_score(targs_arr,preds_arr, average=None)
        scores_keys = target_df.columns[mask].tolist()
        scores_dict = {k:v for (k,v) in zip(scores_keys, scores_vals)}
    else:
        try:
            scores_dict = {'mean': average_precision_score(targs_arr,preds_arr, average=average)} 
        except:
            scores_dict = {'mean': 0}     
    return scores_dict['mean']


#https://github.com/Project-MONAI/tutorials/blob/main/modules/3d_image_transforms.ipynb
#Note that MONAI transformations work on both the label and the image, intended for segmentation masks

class Augmentation():
    '''Vol Augmentation to work with 3d tensors Outputting (z,y,x)
       Img Augmentation to work with numpy arrays, initial shape (20,20,3) or similar
       Outputing (3,16,16) as a PyTorch Tensor (C,H,W)'''

    def _slice_one_dim(self, array, slice_pattern):
        """
        Apply a slice or indices dynamically to the first or second dimension of a 3D array.
        slice_pattern is a tuple (the pattern (array, list of slice), the dimension (0 or 1)
        """
        pattern= slice_pattern[0]
        dim = slice_pattern[1]
        if isinstance(pattern, (slice, list, np.ndarray)):
            if dim==0:
                return array[pattern]
            else: 
                return array[:, pattern]
        else:
            raise ValueError("Slice pattern must be a slice, list, or numpy array.")

    def __init__(self, mean, std, height, width, layers):
        self.mean = mean
        self.std = std
        #self.to_trim = (layers -12) // 2   #eg (40-12) //2 = 14  so np.arrange((14, 36), 0)
        self.to_trim = 13

        #for 9 patches
        #self.slices = [(np.arange(3, 30), 0),
        #               (np.arange(4, 31), 0),
        #               (np.arange(5, 32), 0),
        #               (np.arange(6, 33), 0),
        #               (np.arange(7, 34), 0),
        #               (np.arange(8, 35), 0),
        #               (np.arange(9, 36), 0),
        #               ]

        #four patches about central 12 slices
        '''
        self.slices = [(np.arange(self.to_trim - 3, self.to_trim + 9), 0),
                       (np.arange(self.to_trim - 2, self.to_trim +10), 0),
                       (np.arange(self.to_trim - 1, self.to_trim +11), 0),
                       (np.arange(self.to_trim,     self.to_trim +12), 0),
                       (np.arange(self.to_trim + 1, self.to_trim +13), 0),
                       (np.arange(self.to_trim + 2, self.to_trim +14), 0),
                       (np.arange(self.to_trim + 3, self.to_trim +15), 0),
                       ]
        '''
        #four patches about central 24 slices taking every second
        self.slices = [
                       (np.arange(5, 29, 2),0),
                       (np.arange(6, 30, 2),0),
                       (np.arange(7, 31, 2),0),
                       (np.arange(8, 32, 2),0),
                       (np.arange(9, 33, 2),0),
                       (np.arange(10, 34, 2),0),
                       (np.arange(11, 35, 2),0),
                       ]


        self.vol_train = Compose([
            RandRotate90(spatial_axes=[0, 1], max_k=3),  
                        OneOf([Lambda(func=lambda x: self._slice_one_dim(x, s)) for s in self.slices]),
                        RandSpatialCrop(roi_size=(32, 32), random_center=True, random_size=False),
            RandFlip(prob=0.5, spatial_axis=0),  #Spoils learning from handedness (chirality)
            RandGaussianNoise(prob=0.3),
            RandAdjustContrast(prob=0.3),

        ])

        self.vol_val_tta_0 = Compose([
                                Lambda(func = lambda x: self._slice_one_dim(x, self.slices[3])),
                                RandSpatialCrop(roi_size=(32, 32), random_center=False, random_size=False)
                                ])

        self.vol_val_tta_1 = Compose([
                                RandFlip(prob=1, spatial_axis=0),
                                #RandFlip(prob=1, spatial_axis=1),
                                RandRotate90(prob=1, spatial_axes=[0, 1], max_k=1),
                                RandRotate90(prob=1, spatial_axes=[0, 1], max_k=1),
                                Lambda(func = lambda x: self._slice_one_dim(x, self.slices[3])),
                                RandSpatialCrop(roi_size=(32, 32), random_center=False, random_size=False)
                                ])

        self.vol_val_tta_2 = Compose([
                                    RandRotate90(prob=1, spatial_axes=[0, 1], max_k=1),
                                    Lambda(func = lambda x: self._slice_one_dim(x, self.slices[3])),
                                    RandSpatialCrop(roi_size=(32, 32), random_center=False, random_size=False)
                                ])

        self.img_train = A.Compose([
                    A.CenterCrop(height=2*height, width=2*width, p=1),
                    A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1),
                    ToTensorV2()])

        self.img_val = A.Compose([
            A.CenterCrop(height=2*height, width=2*width, p=1),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1), 
            ToTensorV2()]) #Note: ToTensorV2 turns HWC Numpy to CHW Tensor


class TomoDataset(Dataset):
    def __init__(self,
                 df,
                 vol_transform=None,
                 img_transform=None,
                 num_patches=2,
                 random_flip=False):
        self.df = df
        self.vol_transform = vol_transform
        self.img_transform = img_transform
        self.num_patches = num_patches
        self.flip = random_flip

    def __len__(self):
        return len(self.df)


    def reshape_tomo(self, array):
        """
        Starting with an array shaped (z, y, x) (Slices, Height, Width):
        - Crop so the total slices is exactly 12.
        - Break into 4 parts along the slice direction.
        - Re-assemble the 4 parts so the assembled array is (3, 32, 32).
        - Transpose, so it is left as (Height, Width, Channels) to be treated as an image.
        """

        #array = array[14:-14, :, :]  # Adjust indices if necessary
        
        #Tried this, Run)07, it didn't help.
        #if self.flip and random.choice([True, False]):  
        #    array = array.flip(0)

        z_slices = array.shape[0]
        assert z_slices % self.num_patches == 0, "The number of slices after cropping must be divisible by 4"
        parts = np.array_split(array, self.num_patches, axis=0)

        if self.num_patches == 4:
            top = np.concatenate(parts[:2], axis=2)  # Concatenate the first two along width
            bottom = np.concatenate(parts[2:], axis=2)  # Concatenate the last two along width
            assembled_array = np.concatenate([top, bottom], axis=1)  # Concatenate along height
        else:
            top = np.concatenate(parts[:3], axis=2)
            middle = np.concatenate(parts[3:6], axis=2)  
            bottom = np.concatenate(parts[6:], axis=2)
            assembled_array = np.concatenate([top, middle, bottom], axis=1)  # Concatenate along height
        hwc_array = np.transpose(assembled_array, (1, 2, 0))  # Transpose to (y, x, z)
        return hwc_array


    def __getitem__(self, index):
        while True:
            row = self.df.iloc[index]
            f_pth = row['File_Path']
            tomo = np.load(f_pth)

            if tomo is not None:
                break
            print(f"Warning: Unable to load the file at '{f_pth}'. Skipping...")
            index = torch.randint(0, len(self.df), (1,)).item()  # Get a random index

        #tomo = np.transpose(tomo, (1, 0, 2))
        

        tomo = self.vol_transform(tomo) #Returns a cube of dimension (12,32,32)
        image = self.reshape_tomo(tomo) #Numpy array like an image, of shape (H, W, 3)  dims (3, 64,64)
        min_vals = np.min(image)
        max_vals = np.max(image) 
        image = (image - min_vals) / (max_vals - min_vals)  #Min-Max normalise on [0,1]
        image_tensor = self.img_transform(image=image)['image']  #Tensor with channels first (3, H, W)
        ohe_vals = row.iloc[2:].values.astype(int)
        targets = torch.from_numpy(ohe_vals).float()
        return image_tensor, targets


class BasicHead(nn.Module):
    '''Bare bones fc classifier head'''
    def __init__(self, in_features, num_classes):
        super(BasicHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomModel(pl.LightningModule):
    '''Pytorch Lightning Model with customised callbacks, logging and loss functions'''

    def _randomize_last_n_layers(self, n=15):
                layers = list(self.backbone.named_modules())[-n:]  # Get last `n` layers

                for name, module in layers:
                    if hasattr(module, 'reset_parameters'):  # Reset weights if the layer supports it
                        module.reset_parameters()
                    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):  
                        # Manually reinitialize if no `reset_parameters` method
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)


    def __init__(self,
                class_list,
                loss = nn.CrossEntropyLoss(),
                lr = 1e-3,
                weight_decay = 1e-5,
                unfreeze_0=None,
                unfreeze_1=None,
                unfreeze_lyrs_0=2,
                unfreeze_lyrs_1=6,
                mixup_alpha = 0.5,
                use_mixup = False,
                model_name = 'efficientnetv2_l_21k',
                initial_lr=  1e-5,
                warmup_epochs= 2,
                cycle_length= 6,
                min_lr= 1e-5,
                lr_decay= .5,
                batch_size=64,
                activation='Softmax',
                randomise_top=False,
                num_randomised=0
                ):
        super().__init__()

        self.unfreeze_0 = unfreeze_0
        self.unfreeze_1 = unfreeze_1
        self.unfreeze_layers_0 = unfreeze_lyrs_0
        self.unfreeze_layers_1 = unfreeze_lyrs_1
        self.lr = lr
        self.decay = weight_decay
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.activation = activation

        for param in self.backbone.parameters():
            param.requires_grad = False 

        if randomise_top:
            self._randomize_last_n_layers(n=num_randomised)
        
        unfrozen_layers = list(self.backbone.children())[-num_randomised:]
        for layer in unfrozen_layers:
            if not isinstance(layer, nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = True

        if model_name.startswith('eca_nfnet') or model_name.startswith('convnext'):
            self.in_features = self.backbone.head.fc.in_features
            setattr(self.backbone.head, 'fc', BasicHead(self.in_features, self.num_classes))
        elif model_name.startswith('efficientvit'): #or model_name.startswith(deit_tiny):
            for layer in self.backbone.head.classifier:
                if isinstance(layer, torch.nn.Linear):
                    self.in_features = layer.in_features
                    break
            setattr(self.backbone.head, 'classifier', BasicHead(self.in_features, self.num_classes))
        elif model_name.startswith('mobilevit'):
            self.in_features = self.backbone.head.fc.in_features
            setattr(self.backbone.head, 'fc', BasicHead(self.in_features, self.num_classes))
        elif model_name.startswith('deit_tiny'):
            self.in_features = self.backbone.head.in_features
            setattr(self.backbone.head, 'linear', BasicHead(self.in_features, self.num_classes))
        elif hasattr(self.backbone, 'fc'):
            self.in_features = self.backbone.fc.in_features
            setattr(self.backbone, 'fc', BasicHead(self.in_features, self.num_classes))
        elif hasattr(self.backbone, 'classifier'):
            self.in_features = self.backbone.classifier.in_features
            setattr(self.backbone, 'classifier', BasicHead(self.in_features, self.num_classes))
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'linear'):
            self.in_features = self.backbone.head.linear.in_features
            setattr(self.backbone, 'linear', BasicHead(self.in_features, self.num_classes))
        else:
            raise AttributeError("The backbone does not have a 'fc' or 'classifier' or 'head', layer. Update the code to handle the correct layer name.")

        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        self.mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.loss_function = loss
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.lr_decay = lr_decay
        self.batch_size = batch_size


    def forward(self, images):
        logits = self.backbone(images)
        return logits


    def configure_optimizers(self):
        def custom_lr_scheduler(epoch):
            '''CosineAnealingWarmRestarts but with a decay and a warmup'''
            initial = self.initial_lr / self.lr
            rel_min = self.min_lr / self.lr
            step_size = (1-initial) / self.warmup_epochs
            warmup = initial + step_size * epoch if epoch <= self.warmup_epochs else 1
            cycle = epoch-self.warmup_epochs
            decay = 1 if epoch <= self.warmup_epochs else self.lr_decay ** (cycle // self.cycle_length)
            phase = np.pi * (cycle % self.cycle_length) / self.cycle_length
            cos_anneal = 1 if epoch <= self.warmup_epochs else  rel_min + (1 - rel_min) * (1 + np.cos(phase)) / 2
            return warmup * decay * cos_anneal #this value gets multipleid by the initial lr (self.lr)

        #optimizer = Adam(self.parameters(), lr=self.lr)
        #optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), nesterov=True)
        #optimizer = SGD(self.parameters(), lr=self.lr, momentum=.8)

        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.02)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, target = batch
        y_pred = self(image)
        loss = self.loss_function(y_pred,target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        train_output = {"train_loss": loss, "logits": y_pred, "targets": target}
        self.train_outputs.append(train_output)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, batch_size=self.batch_size)
        output = {"val_loss": val_loss, "logits": y_pred, "targets": target}
        self.val_outputs.append(output)
        return {"val_loss": val_loss, "logits": y_pred, "targets": target}

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def on_validation_epoch_end(self):
        print('validation epoch end')
        val_outputs = self.val_outputs
        avg_val_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean().cpu().detach().numpy()
        output_val_logits = torch.cat([x['logits'] for x in val_outputs],dim=0)
        val_targets = torch.cat([x['targets'] for x in val_outputs],dim=0).cpu().detach().numpy()

        train_outputs = self.train_outputs
        if train_outputs:
            train_losses = [x['train_loss'].cpu().detach().numpy() for x in train_outputs]
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
            output_train_logits = torch.cat([x['logits'] for x in train_outputs],dim=0)
            train_targets = torch.cat([x['targets'] for x in train_outputs],dim=0).cpu().detach().numpy()
        else:
            avg_train_loss = avg_val_loss #we need this because the first time it's an empty list
            output_train_logits = torch.ones(1,output_val_logits.shape[1])
            train_targets = torch.zeros(1, output_val_logits.shape[1])

        if self.activation == 'Softmax':
            val_probs = F.softmax(output_val_logits, dim=1).cpu().detach().numpy()
            train_probs = F.softmax(output_train_logits, dim=1).cpu().detach().numpy()
        else:
            val_probs = F.sigmoid(output_val_logits).cpu().detach().numpy()
            train_probs = F.sigmoid(output_train_logits).cpu().detach().numpy()

        val_pred_df = pd.DataFrame(val_probs, columns = self.class_list)
        val_target_df = pd.DataFrame(val_targets, columns = self.class_list)
        train_pred_df = pd.DataFrame(train_probs, columns = self.class_list)
        train_target_df = pd.DataFrame(train_targets, columns = self.class_list)

        train_cmap = get_map_score(train_target_df, train_pred_df) if len(train_target_df) > 16 else 1
        val_cmap = get_map_score(val_target_df, val_pred_df) if len(train_target_df) > 16 else 1

        self.metrics_list.append({'train_loss':avg_train_loss,
                                  'val_loss': avg_val_loss, 
                                  'train_cmap': train_cmap,
                                  'val_cmap': val_cmap, 
                                  })

        print(f'epoch {self.current_epoch} train loss {avg_train_loss}')
        print(Colour.S + f'epoch {self.current_epoch} validation loss: ' + Colour.E, avg_val_loss)
        print(Colour.S +f'epoch {self.current_epoch} validation mAP score: ' + Colour.E, val_cmap)
        optimizer_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f'Learning rate from optimiser at epoch {self.current_epoch}: {optimizer_lr}')

        self.val_outputs = []
        self.train_outputs = []
        self.val_epoch +=1

    def on_train_epoch_end(self, *args, **kwargs):
        print('train epoch end')
        if (self.unfreeze_0 is not None) and (self.current_epoch == self.unfreeze_0):
            unfrozen_layers = list(self.backbone.children())[-self.unfreeze_layers_0:]
            for layer in unfrozen_layers:
                if not isinstance(layer, nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = True
            print(Colour.S + f'Unfreezing the top {self.unfreeze_layers_0} '
            f'layers of the backbone after {self.current_epoch} epochs' + Colour.E)
        elif (self.unfreeze_1 is not None) and (self.current_epoch == self.unfreeze_1):
            unfrozen_layers = list(self.backbone.children())[-self.unfreeze_layers_1:]
            for layer in unfrozen_layers:
                if not isinstance(layer, nn.BatchNorm2d):
                    for param in layer.parameters():
                        param.requires_grad = True
            print(Colour.S + f'Unfreezing the top {self.unfreeze_layers_1} '
            f'layers of the backbone after {self.current_epoch} epochs' + Colour.E)


    def get_my_metrics_list(self):
        return self.metrics_list


def get_dataloaders(df_train,
                    df_valid,
                    transforms,
                    num_workers=0,
                    num_patches=2,
                    batch_size=64,
                    sample_weights=None,
                    epoch_length=1000000):

    ds_train = TomoDataset(df_train, transforms.vol_train, transforms.img_train, num_patches=num_patches, random_flip=True)
    ds_val = TomoDataset(df_valid, transforms.vol_val_tta_0, transforms.img_val, num_patches=num_patches, random_flip=False)

    p_workers = True if num_workers > 0 else False
    if sample_weights is not None:
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=epoch_length)
        dl_train = DataLoader(ds_train,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=num_workers)
    else:
        dl_train = DataLoader(ds_train,
                              batch_size=batch_size,
                              persistent_workers=p_workers,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    dl_val = DataLoader(ds_val,
                        batch_size=batch_size,
                        persistent_workers=p_workers,
                        num_workers = num_workers,
                        )
    return dl_train, dl_val, ds_train, ds_val


class EarlyStoppingMinEpochs(EarlyStopping):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            print('Not checking for early stopping check because start_epoch not yet reached')
            return
        # Call the parent method to retain the usual early stopping logic
        print('Checking for early stopping')
        super().on_validation_end(trainer, pl_module)


def run_training(weights_dir,
                 dl_train,
                 dl_val,
                 logger=None,
                 epochs=16,
                 fixed_epochs=None,
                 patience=4,
                 loss_function = nn.CrossEntropyLoss(),
                 model=None
                 ):
    '''Function to instantiate trainer, run training
    returns training metrics and path to the last weights'''

    print("Running training...")
    early_stop_callback = EarlyStoppingMinEpochs(monitor="val_loss",
                                                 start_epoch=10,
                                                 min_delta=0,
                                                 patience=patience,
                                                 verbose= True,
                                                 mode="min")

    # saves top- checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=8,
                                          monitor="val_loss",
                                          mode="min",
                                          dirpath=weights_dir,
                                          save_last= True,
                                          save_weights_only=True,
                                          verbose= True,
                                         )
    
    callbacks_to_use = [checkpoint_callback]
    
    if fixed_epochs is not None:
        epochs = fixed_epochs
    if fixed_epochs is None:
        callbacks_to_use = callbacks_to_use + [early_stop_callback]

    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps = 23,
        callbacks=callbacks_to_use,
        gradient_clip_val=1.0,  # Set the max norm value for gradient clipping
        precision='16-mixed',
        accelerator='gpu')

    print("Running trainer.fit")
    trainer.fit(model, train_dataloaders = dl_train, val_dataloaders = dl_val)
    best_model_pth = trainer.checkpoint_callback.best_model_path
    metrics = model.get_my_metrics_list()
    best_epoch = trainer.current_epoch - patience
    del model, trainer, loss_function, dl_train, dl_val
    return metrics, best_model_pth, best_epoch


def get_best_model(class_list, model_pth, model_name, activation):
    '''Loads a model and sets up for evaluation'''
    print(f'using model {model_name} for evaluation')
    best_model_state_dict = torch.load(model_pth)['state_dict']

    unwanted_prefixes = ['loss_function.weight']

    filtered_state_dict = {
        key: value for key, value in best_model_state_dict .items()
        if not any(key.startswith(prefix) for prefix in unwanted_prefixes)
    }
    best_model = CustomModel(class_list, model_name=model_name, activation=activation)
    best_model.load_state_dict(filtered_state_dict)
    best_model.eval()
    return best_model


def check_best_model(best_model,
                    test_df,
                    use_gpu,
                    transforms,
                    activation='Sigmoid',
                    batch_size=8,
                    num_patches=4
                    ):
    '''Checks the model, and runs a small sample of images as a quick sanity check
       Runs over all tty combinations, and returns a list of dataframes, one from each
    '''

    def probs_to_df(probs, targs):
        results = []
        for row, target in zip(probs, targs):
            target = np.argmax(target)
            class_name = class_names[target]
            class_scores = {name: prob for name, prob in zip(class_names, row)}
            result = {'Targets':class_name}
            results.append(result | class_scores)
        return pd.DataFrame(results)
    
    class_names = ['apo-ferritin',
                   'beta-amylase',
                   'beta-galactosidase',
                   #'empty',
                   'ribosome',
                   'thyroglobulin',
                   'virus-like-particle']

    test_ds_0 = TomoDataset(test_df, transforms.vol_val_tta_0, transforms.img_val, num_patches=num_patches, random_flip=False)
    test_ds_1 = TomoDataset(test_df, transforms.vol_val_tta_1, transforms.img_val,  num_patches=num_patches, random_flip=False)  # Flip on 3 axis
    #test_ds_2 = TomoDataset(test_df, transforms.vol_val_tta_2, transforms.img_val,  num_patches=num_patches, random_flip=False)  # Rotate 90

    tty_dfs = []
    for idx, ds in enumerate([test_ds_0, test_ds_1]):
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
        print(f'Evaluating {batch_size * len(loader)} example test images with tta_{idx}')

        all_probs=[]
        all_targets=[]
        for images, targets in loader:
            if use_gpu:
                images, targets, best_model = images.cuda(), targets.cuda(), best_model.cuda()
            logits = best_model(images)
        
            if activation == 'Softmax':
                probs = F.softmax(logits, dim=1)
            else:
                probs = F.sigmoid(logits)

            probs_npy = probs.view(-1, probs.size(-1)).detach().cpu().numpy()
            targs_npy = targets.view(-1, probs.size(-1)).detach().cpu().numpy()
            all_probs.append(probs_npy) #A flat list of probabilites for the whole tty loop
            all_targets.append(targs_npy)
        concat_probs = np.concatenate(all_probs, axis=0)
        concat_targs = np.concatenate(all_targets, axis=0)
        tty_dfs.append(probs_to_df(concat_probs, concat_targs))
        del logits, probs
    return tty_dfs


def evaluate_performance(df_list, class_names, thresholds, print_result=True):
    '''Generate performance metrics from a list of results dataframes, and also from
    an ensemble of those dataframes.  Formate 'Targets', then class names as col headers'''

    total_samples = len(df_list[0])
    for df in df_list:
        df = df.copy()
        results = []
        for class_name in class_names:
            p_dict = {}
            threshold = thresholds[class_name]
            df[class_name] = (df[class_name] >= threshold).astype(int)

            tp = df[(df['Targets'] == class_name) & (df[class_name] == 1)].shape[0]
            tn = df[(df['Targets'] != class_name) & (df[class_name] == 0)].shape[0]
            fp = df[(df['Targets'] != class_name) & (df[class_name] == 1)].shape[0]
            fn = df[(df['Targets'] == class_name) & (df[class_name] == 0)].shape[0]

            p_dict['Class'] = class_name
            p_dict['Accuracy'] = (tp + tn) / total_samples
            p_dict['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 0
            p_dict['Recall'] = tp / (tp + fn) if (tp + fn) != 0 else 0
            results.append(p_dict)

        df_performance = pd.DataFrame(results).set_index('Class').round(3)
        if print_result:
            print(df_performance,'\n')
    return df_performance


def plot_train_metrics(metrics, save_path):
    '''Saves a plot of the training metrics for later analysis'''
    #The first check is at 0, second at 0.5.
    train_losses = [x['train_loss'] for x in metrics][1:]
    val_losses = [x['val_loss'] for x in metrics][1:]
    train_precision = [x['train_cmap'] for x in metrics][1:]
    val_precision = [x['val_cmap'] for x in metrics][1:]
    num_checks = len(val_losses) + 1  #+1 because the list was sliced
    print(f'There were {num_checks} checkpoints recorded')
    time_axis = [0.5*x - 0.5 for x in range(2, num_checks+1)]

    _, ax = plt.subplots()
    plt.plot(time_axis, train_losses, 'r', label='Train Loss')
    plt.plot(time_axis, val_losses, '--k', label='Val Loss')
    plt.legend()
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.tick_params('both', colors='r')

    # Get second axis
    ax2 = ax.twinx()
    plt.plot(time_axis, train_precision, 'b', label='Train mAP')
    plt.plot(time_axis, val_precision, '--g', label='Val mAP')
    ax2.set_ylabel('Accuracy')
    plt.legend()
    plt.legend(loc='lower left')
    ax.tick_params('both', colors='b')
    plt.savefig(save_path)


class FocalLoss(nn.Module):
    '''Multi-class Focal loss with pre-computed values for alpha 
       assumes one-hot encoded targets'''

    def __init__(self, alphas, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alphas = torch.FloatTensor(alphas)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = torch.sum(targets * self.alphas, dim=1)
        loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


# https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
class BCEFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5, gamma=3):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = torch.FloatTensor(class_weights).to('cuda')
        print(f'hello, inside bce, self.gamma is {self.gamma}  self.class_weights is {self.class_weights}')

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)
        focal_weight = targets * (1 - probas) ** self.gamma + (1 - targets) * probas ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = self.class_weights * focal_weight * alpha_weight * bce_loss
        return loss.mean()


def get_loss_function(name, gamma, class_weights):
    '''Select and instantiate a choice of loss function'''

    if name == 'FocalLoss':
        class_weights = torch.FloatTensor(class_weights)
        loss = FocalLoss(class_weights, gamma)
        activation = 'Softmax'
    if name == 'BinaryFocalLoss':
        loss = BCEFocalLoss(class_weights, alpha=0.7, gamma=gamma)
        activation = 'Sigmoid'
    else:
        #torch.tensor(class_weights).to('cuda')
        #loss = nn.CrossEntropyLoss(weight=class_weights)
        loss = nn.CrossEntropyLoss()
        activation = 'Softmax'
    return activation, loss


def get_training_data(fldr_pth):
    """Build a labels dataframe from the file structure of the training images"""
    npy_files = {path: path.parent.name for path in fldr_pth.rglob('*.npy')}
    df = pd.DataFrame(list(npy_files.items()), columns=["File_Path", "Targets"])
    df = df[df['Targets'] != 'empty']
    print(df.head())
    return df


def split_data(df, val_fraction=0.1, folds=1):
    print(df.head())
    targets_list = list(map(str, list(df['Targets'].unique())))
    targets_list.sort()
    num_classes = len(targets_list)
    print(f'There are {len(df)}, total images in labels dataframe')
    print(f'There are a total of {num_classes} classes')

    total_images = len(df)
    indices = list(range(total_images))
    random.shuffle(indices)

    all_folds = []
    num_val = int(np.floor(val_fraction * len(indices))) 
    val_starts = [i * num_val for i in range(folds)]
    for val_start in val_starts:
        val_idx = indices[val_start:val_start + num_val]
        train_idx = list(set(indices) - set(val_idx))
        val_data = df.iloc[val_idx].copy()
        train_data = df.iloc[train_idx].copy()
        all_folds.append([train_data, val_data])

    print(' Training set size: \t', len(train_idx))
    print(' Validation set size: \t', len(val_idx))
    print(' Total dataset: \t', total_images)
    print(f'The target list is: {targets_list}')

    return all_folds


def encode_df(df, class_list):
    df = pd.concat([df, pd.get_dummies(df['Targets'], dtype=int)], axis=1)
    df = df[['Targets','File_Path'] + class_list] ## Ensure all dataframes have columns in the same order
    return df


def plot_two_distributions(dist1, 
                           dist2, 
                           save_path,
                           label1='Distribution 1',
                           label2='Distribution 2',
                           x_max=None, 
                           y_max=None,
                           bins=None):
    
    plt.figure(figsize=(6, 4))
    ax = sns.histplot(dist1, bins=bins, kde=False, color='blue', label=label1, stat='density')
    sns.histplot(dist2, bins=bins, kde=False, color='orange', label=label2, stat='density', ax=ax)
    plt.title(f'{label1} and {label2} score distributions')
    plt.xlabel('Value')
    ax.set(xlim=(0, x_max) if x_max is not None else None)
    ax.set(ylim=(0, y_max) if y_max is not None else None)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot
    plt.close()


def ensemble_ttas(results_list):
    df_0 = results_list[0]
    targets_column = df_0['Targets']
    columns = df_0.iloc[:, 1:].columns
    arrays = np.array([df.iloc[:, 1:].values for df in results_list])
    mean_array = arrays.mean(axis=0)
    mean_df = pd.DataFrame(mean_array, columns=columns, index=df_0.index)
    mean_df.insert(0, 'Targets', targets_column)
    return mean_df


def plot_score_distributions(df, class_names, plot_dir_path):
    '''For a particular class, will save a plot showing the probability scores
    for all true positives and true negatives
    arguments: class name (string) df with a 'Targets' col (str) 
               and cols of probabilities with class names as headers'''
    print(df.head())
    for class_name in class_names:
        path = plot_dir_path / f'{class_name}_distribution.jpg'
        df_subset=df.copy()
        df_subset = df_subset[[class_name, 'Targets']]

        positive_vals = df[df['Targets']==class_name][class_name].values
        negative_vals = df[df['Targets']!=class_name][class_name].values

        plot_two_distributions(positive_vals,
                        negative_vals,
                        path,
                        label1=f'True Positive {class_name}',
                        label2=f'True Negative {class_name}',
                        bins=40,
                        x_max=1,
                        y_max=40)
    return

def compute_weighted_f4(df, weights):
    beta = 4
    weights = list(weights.values())
    precision = df['Precision']
    recall = df['Recall']
    df['f_beta'] = (1+beta**2) * (precision * recall) / (beta**2 * precision + recall)
    df['weight'] = weights
    df['weighted_fb'] = df['f_beta'] * df['weight'] / sum(weights)
    weighted_f4 = df['weighted_fb'].sum()
    #print(df)
    return weighted_f4


def train_one_fold(fold,
                    data_cfg,
                    paths,
                    train_cfg,
                    eval_cfg,
                    image_cfg,
                    loss,
                    activation,
                    augmentations,
                    num_workers,
                    plot_metrics=False,
                    max_epochs=None,
                    model_weights_path=None,
                    fixed_epochs=None
                    ):

    logger = CSVLogger(save_dir=paths.results_dir, name='csv_logger')
    training_model = CustomModel(data_cfg.CLASS_NAMES,
                    loss=loss,
                    lr = train_cfg.LEARNING_RATE,
                    weight_decay=train_cfg.WEIGHT_DECAY,
                    unfreeze_0=train_cfg.EPOCHS_BACKBONE_FROZEN_0,
                    unfreeze_lyrs_0=train_cfg.UNFREEZE_LAYERS_0,
                    unfreeze_1=train_cfg.EPOCHS_BACKBONE_FROZEN_1,
                    unfreeze_lyrs_1=train_cfg.UNFREEZE_LAYERS_1,
                    mixup_alpha = train_cfg.MIXUP_ALPHA,
                    use_mixup=train_cfg.USE_MIXUP,
                    model_name=train_cfg.MODEL_NAME,
                    initial_lr=train_cfg.INITIAL_LR,
                    warmup_epochs=train_cfg.WARMUP_EPOCHS,
                    cycle_length=train_cfg.LR_CYCLE_LENGTH,
                    min_lr=train_cfg.MIN_LR,
                    lr_decay=train_cfg.LR_DECAY,
                    batch_size=train_cfg.BATCH_SIZE,
                    activation=activation,
                    randomise_top=train_cfg.RANDOMISE_TOP,
                    num_randomised=train_cfg.NUM_RANDOMISED)

    if model_weights_path is not None:
        print('Loading locally trained weights')
        best_model_state_dict = torch.load(model_weights_path)['state_dict']
        unwanted_prefixes = ['loss_function.weight']
        filtered_state_dict = {
            key: value for key, value in best_model_state_dict .items()
            if not any(key.startswith(prefix) for prefix in unwanted_prefixes)
        }
        training_model.load_state_dict(filtered_state_dict)


    train_df, val_df = [encode_df(_df, data_cfg.CLASS_NAMES) for _df in fold]
    dl_train, dl_val, _, _ = get_dataloaders(train_df,
                                            val_df,
                                            augmentations,
                                            num_workers=num_workers,
                                            num_patches=image_cfg.NUM_PATCHES,
                                            batch_size=train_cfg.TRAIN_BATCH_SIZE,
                                            )

    metrics, best_model_path, best_epoch = run_training(paths.weights_pth,
                                    dl_train,
                                    dl_val,
                                    logger=logger,
                                    epochs = max_epochs,
                                    fixed_epochs = fixed_epochs,
                                    patience=train_cfg.PATIENCE,
                                    loss_function= loss,
                                    model=training_model)

    if plot_metrics:
        if model_weights_path is None:
            #This is the case for pre-training
            plot_train_metrics(metrics, paths.pretrain_metrics_pth)
        else:
            #This is the case for fine tuning
            plot_train_metrics(metrics, paths.train_metrics_pth)

    best_model = get_best_model(data_cfg.CLASS_NAMES,
                                best_model_path,
                                train_cfg.MODEL_NAME,
                                activation=activation)
    
    torch.save(best_model.state_dict(), paths.final_weights_pth)
    print(f'Final model saved to {paths.final_weights_pth}')

    results = check_best_model(best_model,
                    val_df,
                    True,    #Use GPU
                    augmentations,
                    batch_size=train_cfg.BATCH_SIZE,
                    activation=activation,
                    num_patches=image_cfg.NUM_PATCHES
                    )
    del best_model

    _ = evaluate_performance(results, data_cfg.CLASS_NAMES, eval_cfg.THRESHOLDS)
    ensemble_df = ensemble_ttas(results)
    precision_recall = evaluate_performance([ensemble_df], data_cfg.CLASS_NAMES, eval_cfg.THRESHOLDS)

    if plot_metrics:
        plot_score_distributions(ensemble_df, data_cfg.CLASS_NAMES, paths.plots_dir)
    weighted_f4 = compute_weighted_f4(precision_recall, eval_cfg.WEIGHTS)

    print(f"The weighted f-beta score from TTA ensemble is {weighted_f4:.3f}")

    for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
        thresholds = {key: threshold for key in eval_cfg.THRESHOLDS.keys()}
        pr = evaluate_performance([ensemble_df],
                                data_cfg.CLASS_NAMES,
                                thresholds,
                                print_result=False)
        thresh_weighted_f4 = compute_weighted_f4(pr, eval_cfg.WEIGHTS)
        print(f"The weighted f-beta score is {thresh_weighted_f4:.3f} with thresholds {threshold}")

    return weighted_f4, best_epoch, best_model_path


# ----------------------------------- Main  --------------------------------------
# --------------------------------------------------------------------------------
def train():
    train_cfg = TrainConfig()
    pretrain_cfg = PreTrainConfig()
    data_cfg = DataConfig()
    image_cfg = ImageConfig()
    eval_cfg = EvalConfig()
    paths = Paths(train_cfg.EXPERIMENT_NAME, train_cfg.RUN_ID)
    num_workers, accelerator  = set_hardware(train_cfg)
    pl.seed_everything(train_cfg.RANDOM_SEED, workers=True)
    random.seed(train_cfg.RANDOM_SEED)
    np.random.seed(train_cfg.RANDOM_SEED)
    torch.manual_seed(train_cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(train_cfg.RANDOM_SEED)
    set_determinism(train_cfg.RANDOM_SEED)

    augmentations = Augmentation(mean=image_cfg.INPUT_MEAN,
                                        std=image_cfg.INPUT_STD,
                                        height=32,
                                        width=32,
                                        layers=40)


    class_weights = list(data_cfg.CLASS_WEIGHTS.values())

    activation, loss = get_loss_function(name=pretrain_cfg.LOSS_FUNCTION,
                                         gamma = pretrain_cfg.FOCAL_GAMMA,
                                         class_weights=class_weights)

    syn_df = get_training_data(paths.syn_image_dir)
    folds = split_data(syn_df, val_fraction = 0.1, folds=1)

    print(f'the fold is {folds[0]}')

    pretrain_f4, pretrain_best_epoch, pretrain_weights = train_one_fold(folds[0],
                                        data_cfg,
                                        paths,
                                        pretrain_cfg,
                                        eval_cfg,
                                        image_cfg,
                                        loss,
                                        activation,
                                        augmentations,
                                        num_workers,
                                        plot_metrics=True,
                                        max_epochs=pretrain_cfg.MAX_EPOCHS,
                                        fixed_epochs=pretrain_cfg.FIXED_EPOCHS,
                                        model_weights_path=None)

    print(f'The pretraining f4 score was {pretrain_f4}')
    print(f'The pretraining best epoch was {pretrain_best_epoch}')


    ##########Pretrained weights not in use yet.
    activation, loss = get_loss_function(name=train_cfg.LOSS_FUNCTION,
                                         gamma = train_cfg.FOCAL_GAMMA,
                                         class_weights=class_weights)
    in_df  = get_training_data(paths.image_dir)
    folds = split_data(in_df, val_fraction = 0.1, folds=5)

    f4_cvs = []
    epochs = []
    for idx, fold in enumerate(folds):
        f4, best_epoch, _ = train_one_fold(fold,
                                        data_cfg,
                                        paths,
                                        train_cfg,
                                        eval_cfg,
                                        image_cfg,
                                        loss,
                                        activation,
                                        augmentations,
                                        num_workers,
                                        plot_metrics=(idx == 0),
                                        max_epochs=train_cfg.MAX_EPOCHS,
                                        model_weights_path=pretrain_weights)
        f4_cvs.append(round(f4, 3))
        epochs.append(best_epoch)

    print(f'\nThe F4 CV Scores are: {f4_cvs}')
    mean = statistics.mean(f4_cvs)
    std_dev = statistics.stdev(f4_cvs)
    mean_epochs = int(np.ceil(statistics.mean(epochs)))
    max_epochs = int(np.ceil(np.max(epochs)))
    print(f'\nThe epochs used are {epochs}')
    print(f'The mean number of epochs to best was {mean_epochs}')
    print(f'The max number of epochs to best was {max_epochs}')
    print(f'The mean F4 is {mean:.3f}')
    print(f'The standard deviation is {std_dev:.3f}')
    print(f'The mean minus the standard deviation is {mean - std_dev:.3f}\n')

    #Final training on all data, with fixed epochs
    all_data = [in_df, folds[0][1]]
    f4, best_epoch, _ = train_one_fold(all_data,
                                    data_cfg,
                                    paths,
                                    train_cfg,
                                    eval_cfg,
                                    image_cfg,
                                    loss,
                                    activation,
                                    augmentations,
                                    num_workers,
                                    plot_metrics=False,
                                    max_epochs=mean_epochs,
                                    model_weights_path=pretrain_weights)

    print('\nHard Names (Weight 2): thyroglobulin and β-galactosidase')
    print('Easy Names (Weight 1): ribosome, virus-like particles, apo-ferritin')
    print('No score: β-amylase')

    gc.collect()
    torch.cuda.empty_cache()

# ---------------------- Run Training From Default Configuration--------------------------
# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    train()