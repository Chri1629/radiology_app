import os
import pydicom
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import pandas as pd
import math
import imageio
from roipoly import RoiPoly
import matplotlib
matplotlib.use('TkAgg')

def compute_features(self):
    features = pd.DataFrame(np.array([[RefDs.PatientID, volume, surface, spherical__disproportion, sphericity, surfacevolume__ratio]]),
                    columns = ['ID', 'volume', 'surface', 'spherical__disproportion', 'sphericity', 'surfacevolume__ratio'])
    features[['volume', 'surface', 'spherical__disproportion', 'sphericity', 'surfacevolume__ratio']] = features[['volume', 'surface', 'spherical__disproportion', 'sphericity', 'surfacevolume__ratio']].astype("float")
    features = features.round(2)
    return features

def create_circular_mask(h, w, center=None, radius=None):
        if center is None:
            center = (int(w/2), int(h/2))
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

def load_file(PathDicom, fun):
        file_list = os.listdir(PathDicom)
        file_list_sorted = sorted(file_list, key = fun)

        file_list_sorted_dir = []
        for file in file_list_sorted:
            if ".dcm" in file.lower():
                file_list_sorted_dir.append(os.path.join(PathDicom,file))
        return file_list_sorted_dir

def extract_info(info, slices):
    RefDs = pydicom.read_file(info)
    ConstPixelDims = (slices, int(RefDs.Rows), int(RefDs.Columns))
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness)) # VEDERE SE SERVE
    dimx = float(RefDs.PixelSpacing[0])
    dimy = float(RefDs.PixelSpacing[1])
    dimz = float(RefDs.SliceThickness)
    dim_voxel = dimx * dimy * dimz

    return dimx, dimy, dimz, dim_voxel, ConstPixelDims, ConstPixelSpacing, RefDs

def create_array_dicom(file_list_sorted_dir, ConstPixelDims, RefDs):
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        # loop through all the DICOM files
        for filenameDCM in file_list_sorted_dir:
            ds = pydicom.dcmread(filenameDCM)
            img_rescaled = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            ArrayDicom[file_list_sorted_dir.index(filenameDCM), :, :] = img_rescaled

        # Se è una PET aggiungere quelli commentati, se è una CT lasciare i commenti
        #ArrayDicom[ArrayDicom < 0] = np.amax(ArrayDicom)
        ArrayDicom[ArrayDicom == np.amin(ArrayDicom)] = - 1000
        nvoxel = ArrayDicom.shape[1] * ArrayDicom.shape[2]
        minimum = np.amin(ArrayDicom)
        maximum = np.amax(ArrayDicom)
        #stackimg = ArrayDicom - minimum
        #stackimg = stackimg/maximum
        stackimg = ArrayDicom.copy()
        return ArrayDicom, stackimg, nvoxel

def first_4chars(x):
        return(int(x[:-4]))

def last_4chars(x):
    return(x[-8:])

class MedicalPlot:

    def complete_dicom(stackimg_mask, norm, savePath = None):
        plt.style.use('dark_background')
        fig1 = plt.figure(figsize = (30,30))

        for i in range(len(stackimg_mask)):
            fig = plt.subplot(math.ceil(len(stackimg_mask)**(1/2)), math.ceil(len(stackimg_mask)**(1/2)), i+1)
            plt.imshow(stackimg_mask[i], cmap=plt.cm.gray,
                    norm = norm)
            plt.title(f"Slice {i+1}")
            plt.axis('off')
        if savePath is not None:
            fig1.savefig(f"{savePath}/complete.png", dpi = 300, bbox_inches = 'tight')
        plt.show()
        plt.close(fig1)

    def maximum_slices(stackimg_mask, fette_max, norm, savePath = None):
        plt.style.use('dark_background')
        fig1 = plt.figure(figsize = (10,10))
        plt.suptitle("MAXIMUM", fontsize = 30)
        for i, l in enumerate(fette_max):
            fig = plt.subplot(math.ceil(fette_max.shape[0]**(1/2)), math.ceil(fette_max.shape[0]**(1/2)), i+1)
            plt.title(f"Slice {l+1}",fontsize=15)
            plt.imshow(stackimg_mask[l], cmap=plt.cm.gray,
                    norm = norm)
            plt.axis('off')
        if savePath is not None:
            fig1.savefig(f"{savePath}/lesions.png", dpi = 300, bbox_inches = 'tight')
        plt.show()
        plt.close(fig1)

    def create_gif(stackimg_mask, norm, dpi, savePath):
        for i in range(len(stackimg_mask)):
            plt.style.use('dark_background')
            plt.figure(figsize=(6,6))
            plt.imshow(stackimg_mask[i], cmap=plt.cm.gray,
                    norm = norm)
            plt.title(f"Slice {i+1}")
            plt.axis('off')
            path = f'{savePath}/gif'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{i+1}.png", bbox_inches = "tight", dpi = dpi, facecolor="#000000")
            plt.close()
            f_list = os.listdir(path)
            f_list_sorted = sorted(f_list, key = first_4chars)

            # Build GIF
            with imageio.get_writer(f'{savePath}/gif.gif', mode='I') as writer:
                for filename in f_list_sorted:
                    image = imageio.imread(f"{path}/{filename}")
                    writer.append_data(image)


    def manual_crop(s,normalize,n_slice):
        fig = plt.figure(figsize = (15,8))
        plt.imshow(s, interpolation = "nearest", cmap = plt.cm.gray, norm = normalize)
        plt.colorbar()
        plt.title(f"Slice: {n_slice}")
        plt.show(block = False)
        roi1 = RoiPoly(color = 'r', fig = fig)
        plt.close(fig)
        mask = roi1.get_mask(s)
        return s*mask


    def plot_slice(s,normalize,n_slice):
        fig = plt.figure(figsize = (15,8))
        plt.imshow(s, interpolation = "nearest", cmap = plt.cm.gray, norm = normalize)
        plt.colorbar()
        plt.title(f"Slice: {n_slice}")
        plt.show()
        plt.close(fig)


    def plot_masks(masked_slices, n_slice):
        plt.style.use('dark_background')
        fig = plt.figure(figsize = (15,15))

        for i in range(len(masked_slices)):
            plt.subplot(math.ceil(len(masked_slices)**(1/2)), math.ceil(len(masked_slices)**(1/2)), i+1)
            plt.imshow(masked_slices[i], cmap=plt.cm.gray)
            plt.title(f"Slice {n_slice+i}")

        plt.show()
        plt.close(fig)
