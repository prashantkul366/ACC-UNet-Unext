"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        text = sample.get("text", None)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        ###########################################################
        # label = torch.from_numpy(np.array(label, dtype=np.float32))
        # label = (label > 0).float()  # Binarize & ensure float
        ###########################################################     
        sample = {'image': image, 'label': label}

        # UNCOMMENT WHEN MODEL SUPPORTS TEXT 
        # if text is not None:
        #     sample["text"] = text
        # sample = {
        #                 "image": image,
        #                 "label": label,
        #                 "text": text 
        #             }
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        text = sample.get("text", None)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        ###########################################################
        # label = torch.from_numpy(np.array(label, dtype=np.float32))
        # label = (label > 0).float()  # Binarize & ensure float
        ###########################################################
        sample = {'image': image, 'label': label}

        # UNCOMMENT WHEN MODEL SUPPORTS TEXT 
        # sample = {
        #         "image": image,
        #         "label": label,
        #         "text": text   
        #     }
        # if text is not None:
        #     sample["text"] = text    
        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, row_text: dict = None, one_hot_mask: int = False, image_size: int =224, n_labels: int=1) -> None:
        self.dataset_path = dataset_path
        print(f"Dataset path: {dataset_path}")
        self.image_size = image_size        
        # self.input_path = os.path.join(dataset_path, 'img')
        # self.output_path = os.path.join(dataset_path, 'labelcol')
        # self.input_path = os.path.join(dataset_path, 'images')
        # self.output_path = os.path.join(dataset_path, 'masks')
        
        # Option 1: Text style folders
        option1_img = os.path.join(dataset_path, "img")
        option1_mask = os.path.join(dataset_path, "labelcol")

        # Option 2: Generic style folders
        option2_img = os.path.join(dataset_path, "images")
        option2_mask = os.path.join(dataset_path, "masks")


        if os.path.isdir(option1_img) and os.path.isdir(option1_mask):
            # Case 1: img + labelcol
            self.input_path = option1_img
            self.output_path = option1_mask
            print(" Using folders: img/ and labelcol/")

        elif os.path.isdir(option2_img) and os.path.isdir(option2_mask):
            # Case 2: images + masks
            self.input_path = option2_img
            self.output_path = option2_mask
            print(" Using folders: images/ and masks/")

        else:
            # Case 3: Not found
            raise FileNotFoundError(
                f" Dataset folder structure not recognized!\n\n"
                f"Expected either:\n"
                f"  1) img/ and labelcol/\n"
                f"  2) images/ and masks/\n\n"
                f"But found only:\n"
                f"  {os.listdir(dataset_path)}"
            )
        # self.images_list = os.listdir(self.input_path)
        self.images_list = [
                            f for f in os.listdir(self.input_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
                        ]
        self.one_hot_mask = one_hot_mask
        self.n_labels = n_labels
        self.row_text = row_text

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        # return len(os.listdir(self.input_path))
        return len(self.images_list)


    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        if image is None:
            raise ValueError(f" Failed to load image: {os.path.join(self.input_path, image_filename)}")
        # print("img",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image

        #################
        # mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        ################
        # --- Read corresponding mask safely ---
        stem, _ = os.path.splitext(image_filename)

        possible_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        mask_path = None
        for ext in possible_exts:
            candidate = os.path.join(self.output_path, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            raise ValueError(f"❌ No mask found for image: {image_filename} in {self.output_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"⚠️ Mask file exists but could not be read: {mask_path}")

        stem, _ = os.path.splitext(image_filename)
        mask_filename = stem + ".png"

        if self.row_text is not None:
            text = self.row_text.get(mask_filename, "")
        else:
            text = None
        ##########################################################
        # print("mask",image_filename[: -3] + "png")
        # print(np.max(mask), np.min(mask))
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        # print(np.max(mask), np.min(mask))
        if self.n_labels == 1:
            mask[mask<=0] = 0
            mask[mask>0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        assert mask.max() <= 1.0 and mask.min() >= 0.0, f"Mask out of range: {mask.min()} - {mask.max()}"
        sample = {'image': image, 'label': mask}
        # UNCOMMENT WHEN MODEL SUPPORTS TEXT 
        # sample["text"] = text
        
        # sample = {
        #             "image": image,
        #             "label": mask,
        #             "text": text  
        #         }

        if self.joint_transform:
            sample = self.joint_transform(sample)


        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)

        return sample, image_filename