from classes import Patient
from classes import Slices

import pydicom
import os
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import pandas as pd
from utils import *
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import cluster, mixture, metrics

path_dicom = r"C:\Users\matti\Desktop\CT_Positive\manifest-1608266677008\MIDRC-RICORD-1A\MIDRC-RICORD-1A-419639-000082\08-02-2002-CT CHEST WITHOUT CONTRAST-04614\2.000000-ROUTINE CHEST NON-CON-97100"

file_list_sorted_dir = load_file(path_dicom, last_4chars)
dimx, dimy, dimz, dim_voxel, ConstPixelDims, ConstPixelSpacing, RefDs = extract_info(file_list_sorted_dir[0], len(file_list_sorted_dir))
print(dimx)

ArrayDicom, stackimg, nvoxel = create_array_dicom(file_list_sorted_dir, ConstPixelDims, RefDs)

print(ArrayDicom.shape)

#info = {'SUBJECT': RefDs.PatientID, 'SEX': RefDs.PatientSex, 'HEIGHT': float(RefDs.PatientSize), 'WEIGHT': float(RefDs.PatientWeight)}
info = "Ciao"

## creazione oggetto tac
ct = Patient(dimx, dimy, dimz, ArrayDicom.shape[1], ArrayDicom.shape[2], info)

## crea le slice per l'oggetto tac
for i in range(ArrayDicom.shape[0]):
    ct.add_slice(stackimg[i,:,:])

print(ct)

ct.get_sel_slices()

print(ct.sel_slices)
## aggiunge alla lista delle nostre tac
#tacs[sbj][data][tac] = ct
#