import numpy as np
import nibabel as nib
import cv2
import torch

class AAL():
    def __init__(self, aal_dir, labels_dir, rotate=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.rotate = rotate        
    
    def open_aal_axial(self):
        # Load the AAL
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        
        if self.rotate:
            image = np.rot90(image)
        
        return image
    
    def get_data(self):
        image = torch.tensor(self.open_aal_axial().copy()).permute(2, 0, 1)
            
        labels = {}
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                labels[key] = int(value)
        
        return image.float(), labels


class AAL3Channels():
    def __init__(self, aal_dir, labels_dir, transforms=None, rotate=False, duplicate_channels=False):
        self.aal_dir = aal_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.rotate = rotate
        self.duplicate_channels = duplicate_channels
    
    
    def open_scan_sagittal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[11: 71, :, :]
        # For each scan channel
        for i in range(scan.shape[0]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[i, :, :], ((0, 0), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)

        return image
    
    def open_scan_coronal(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, 11: 71, :]
        # For each scan channel
        for i in range(scan.shape[1]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            # WARNING: Use INTER_NEAREST to preserve atlas values!
            image[:, :, i] = cv2.resize(src=scan[:, i, :], dsize=(95, 95), interpolation=cv2.INTER_NEAREST)
            # image[:, :, i] = np.pad(scan[:, i, :], ((8, 8), (8, 8)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                    
        return image
    
    def open_scan_axial(self):
        # Creat an empty array of 95x95x60
        # The first 10 and last 9 channels of original scan are ignored since they don't have useful data.
        image = np.zeros((95, 95, 60))
        # Load the scan
        scan = nib.load(self.aal_dir).get_fdata()[:, :, 11: 71]
        # For each scan channel
        for i in range(scan.shape[2]):
            # Assign the padded scan to image (padding is make the spatial dimension square (95x95))
            image[:, :, i] = np.pad(scan[:, :, i], ((8, 8), (0, 0)))
        
        image = torch.tensor(image.copy()).permute(2, 0, 1)
                
        return image

    def make_grid(self, image):
        # Put "image" channels in a grid-like image
        image_grid = torch.zeros((10 * 95, 6 * 95))
        for row in range(10):
            for col in range(6):
                idx = row * 6 + col
                if idx < 60:
                    image_grid[row * 95: (row * 95) + 95, col * 95: (col * 95) + 95] = image[idx, :, :]
        
        return image_grid
        
    def get_data(self):
        # Open scan at index along sagittal axis
        ch1 = self.open_scan_sagittal()
        # Put channels in a grid-like image
        image_grid_ch1 = self.make_grid(ch1)
        
        # Open scan at index along coronal axis
        ch2 = self.open_scan_coronal()
        # Put channels in a grid-like image
        image_grid_ch2 = self.make_grid(ch2)
        
        # Open scan at index along axial axis
        ch3 = self.open_scan_axial()
        # Put channels in a grid-like image
        image_grid_ch3 = self.make_grid(ch3)
        
        if self.duplicate_channels:
            image_3ch = torch.stack((image_grid_ch2, image_grid_ch2, image_grid_ch2), axis = 0)
        else:
            image_3ch = torch.stack((image_grid_ch1, image_grid_ch2, image_grid_ch3), axis = 0)
            
        if self.rotate:
            image_3ch = torch.rot90(image_3ch, 1, [1, 2])
        
        if self.transforms:
            image_3ch = self.transforms(image_3ch)
        
        
        labels = {}
        with open(self.labels_dir) as file:
            for line in file:
                _, key, value = line.split()
                labels[key] = int(value)
        
        return image_3ch.float(), labels