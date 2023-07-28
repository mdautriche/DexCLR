from pathlib import Path, PurePosixPath
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from skimage.io import imread
import os
import shutil
import numpy as np

class CampDataset(Dataset):
    def __init__(self, root, transform, valid_ratio=0.2, testPhase=False, labels=None, extImgs=[".tif"]):
        # super().__init__()
        self.root = root
        if labels is None:
            self.pretrain = True
        else:
            self.pretrain = False
        self.labels = labels
        self.transform = transform
        self.testPhase = testPhase
        self.valid_ratio = valid_ratio
        self.extImgs = extImgs
        self.images = []
        self.labels = []
        for ext in extImgs:
            self.images = sorted(glob(f'{self.root}/*{ext}'))
            self.labels = sorted(glob(f'{self.labels}/*{ext}'))
            if self.images:
                break
        
        if self.pretrain is False:
            tmp_path = "/".join(self.images[0].split("/")[:2])
            # Divide the dataset and create two new folder: train and test
            if not (os.path.isdir(tmp_path+"/test") and os.path.isdir(tmp_path+"/train")):
                split_dataset_test_train(tmp_path, valid_ratio, self.images, self.labels)
            self.images = []
            self.labels = []
            self.test_images = []
            self.test_labels = []
            for ext in extImgs:
                self.images = sorted(glob(tmp_path+"/train/images/*"+ext))
                self.labels = sorted(glob(tmp_path+"/train/labels/*"+ext))
                self.test_images = sorted(glob(tmp_path+"/test/images/*"+ext))
                self.test_labels = sorted(glob(tmp_path+"/test/labels/*"+ext))
                if self.images:
                    break

            print("Valid_ration = "+str(self.valid_ratio*100))
            self.images, self.labels = train_test_split(self.images, self.labels, valid_ratio=self.valid_ratio)
            print("Real_valid_ratio = "+str(round(len(self.images)/3)))
            
    def __len__(self):
        if self.testPhase is False:
            return len(self.images)
        else:
            return len(self.test_images)
        
    def __getitem__(self, index):
        # normalize
        # convert RGB 8bits
        if self.pretrain is True:
            data = Image.fromarray(rgba2rgb(imread(self.images[index])))
            return self.transform(data)
        else:
            if self.testPhase is False:
                # Basic check if img name and label name are the same
                assert self.images[index].split("/")[-1] == self.labels[index].split("/")[-1], "Finetune: Img diff than label"
                
                data = Image.fromarray(rgba2rgb(imread(self.images[index])))
                # Binary image (1 bands)
                labels = Image.fromarray(imread(self.labels[index]))
                return self.transform(data), self.transform(labels)
            else:
                # Basic check if img and label have the same name
                assert self.test_images[index].split("/")[-1] == self.test_labels[index].split("/")[-1], "Test: Img diff than label"
                
                data = Image.fromarray(rgba2rgb(imread(self.test_images[index])))
                labels = Image.fromarray(imread(self.test_labels[index]))
                return self.transform(data), self.transform(labels)
    
    def setTest(self, testPhase):
            self.testPhase = testPhase


def train_test_split(images, labels, valid_ratio=0.2, createTest=None):
    if createTest is not None:
        valid_ratio = ratio_test_dataset(createTest,len(images))

    assert len(images) == len(labels), 'images and labels in the folder are not the same'
    n = int(len(images)*valid_ratio)
    
    interval = round((len(images)/n)*100)

    sample_inds = list(set([round(x / 100.0) for x in range(0, len(images)*100, interval)]))[:n]
    
    valid_image_samples = [images[idx] for idx in sample_inds if idx < len(images)]
    valid_label_samples = [labels[idx] for idx in sample_inds if idx < len(images)]
    assert len(valid_image_samples) == len(valid_label_samples), 'sample images and sample labels are not the same'
         
    test_images = list(set(images).symmetric_difference(set(valid_image_samples)))
    test_labels = list(set(labels).symmetric_difference(set(valid_label_samples)))
        
    assert len(test_images) == len(test_labels), 'test image and labels are not the same'
    
    if createTest is not None:
        return sorted(test_images), sorted(test_labels), sorted(valid_image_samples), sorted(valid_label_samples)
    else:
        return sorted(valid_image_samples), sorted(valid_label_samples)
    
#########################################################################################################################
def ratio_test_dataset(createTest,len_images,ratio_nb_test=0.60):
    valid_ratio = 1-ratio_nb_test
    assert valid_ratio <= 1, "Error valid ratio superior at 1, Reminder test dataset size should be 1000 images"
    try:
        os.mkdir(createTest+'/test')
        os.mkdir(createTest+'/test/images')
        os.mkdir(createTest+'/test/labels')
    except:
        pass
    try:
        os.mkdir(createTest+'/train')
        os.mkdir(createTest+'/train/images')
        os.mkdir(createTest+'/train/labels')
    except:
        pass
    return valid_ratio

#####################################################################################################################
def split_dataset_test_train(tmp_path, valid_ratio,images, labels):
    test_images, test_labels, images, labels = train_test_split(images, labels, valid_ratio=valid_ratio, createTest=tmp_path)
    for img, label in zip(images, labels):
        shutil.copyfile(img, tmp_path+"/train/images/"+img.split("/")[-1])
        shutil.copyfile(label, tmp_path+"/train/labels/"+label.split("/")[-1])
    for test_img, test_label in zip(test_images, test_labels):
        shutil.copyfile(test_img, tmp_path+"/test/images/"+test_img.split("/")[-1])
        shutil.copyfile(test_label, tmp_path+"/test/labels/"+test_label.split("/")[-1])

def rgba2rgb(rgba):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2]

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return np.asarray( rgb, dtype='uint8' )