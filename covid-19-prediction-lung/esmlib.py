#Warnings filter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#General library
import numpy as np
import matplotlib.pyplot as plt

#Path
import os, os.path
import pickle
import shutil
import glob

# Interactive input
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

#Modeling
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# Scipy
from scipy.misc import face
from scipy import ndimage as ndi
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.segmentation import clear_border, mark_boundaries
from skimage.filters import roberts, sobel
from scipy.signal.signaltools import wiener

#Image
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#Medical Image
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
from roipoly import RoiPoly
import mahotas
import mahotas.demos
import mahotas as mh
import numpy as np
from pylab import imshow, show

# ipywidgets for some interactive plots
from ipywidgets.widgets import * 
import ipywidgets as widgets

# Load the model
pickle_in = open('model/svc_pca_classifier.pickle',"rb")
model = pickle.load(pickle_in)

# Load the PCA object
pickle_in = open('model/pca.pickle',"rb")
pca = pickle.load(pickle_in)

# Function for read and selects three scans from a folder exam
def select_paths(path):
    lista = glob.glob(path + "\*.dcm")
    list_idx = []
    n_dcm = len(lista)
    if n_dcm > 50:
        n_dcm_interval = int(n_dcm / 12)
    else:
        print("The selected folder has less than 50 images.. try to select another folder that contain more scans")
    lista_idx = [n_dcm_interval*4, n_dcm_interval*5, n_dcm_interval*6]
    return lista, lista_idx

# This function extract and print some metadata related to the patient
def get_metadata(folder_path):
    lista = glob.glob(folder_path + "\\" + "*.dcm")
    n = len(lista)
    ds = dcmread(lista[0])
    print("PatientID: " + str(ds.PatientID))
    print("\tSex: " + str(ds.PatientSex))
    print("\tAge: " + str(int(ds.PatientAge[:-1])))
    print("\nStudy Description: " + str(ds.StudyDescription))
    print("\tExam Modality: " + str(ds.Modality))
    print("\tNumber of scans: " + str(len(lista)))
    print("\tImage size " + str(ds.Rows) + "x" + str(ds.Columns))
    print("\tPatient Position: " + str(ds.PatientPosition))
    print("\tSlice Thickness: " + str(ds.SliceThickness))
    print("\tSpace between slices: " + str(ds.SpacingBetweenSlices))
    return None

# This function load and process the image related to the specified path
def load_and_preprocess_dcm(path):
    ds = dcmread(path)
    img = ds.pixel_array
    img = ds.RescaleIntercept + img * ds.RescaleSlope
    minimum = np.amin(np.amin(img))
    maximum = np.amax(np.amax(img))
    img[img == minimum] = -1000
    # Normalization
    img = (img - minimum) / maximum
    new_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    new_img = new_img[50:460,30:480]
    return new_img

# A simple function to show the three images selected in a subplot
def select_and_show_3(lista, lista_idx):
    fig, ([ax1, ax2, ax3]) = plt.subplots(1,3, figsize=(15,12))
    img1 = load_and_preprocess_dcm(lista[lista_idx[0]])
    img2 = load_and_preprocess_dcm(lista[lista_idx[1]])
    img3 = load_and_preprocess_dcm(lista[lista_idx[2]])
    ax1.imshow(img1, cmap=plt.cm.gray)
    ax1.set_title("Scansione n: " + str(lista_idx[0]))
    ax2.imshow(img2, cmap=plt.cm.gray)
    ax2.set_title("Scansione n: " + str(lista_idx[1]))
    ax3.imshow(img3, cmap=plt.cm.gray)
    ax3.set_title("Scansione n: " + str(lista_idx[2]))
    return [img1, img2, img3]

# A simple function to show the three mask in a subplot
def show_3_mask(mask_list, idx_sel):
    fig, ([ax1, ax2, ax3]) = plt.subplots(1,3, figsize=(15,12))
    ax1.imshow(mask_list[0], cmap=plt.cm.gray)
    ax1.set_title("Maschera della scansione n: " + str(idx_sel[0]))
    ax2.imshow(mask_list[1], cmap=plt.cm.gray)
    ax2.set_title("Maschera della scansione n: " + str(idx_sel[1]))
    ax3.imshow(mask_list[2], cmap=plt.cm.gray)
    ax3.set_title("Maschera della scansione n: " + str(idx_sel[2]))
    return None

# This function allow to create a lung mask with the kmeans clustering algorithm
def mask_kmeans(img_array):
    img_reshaped = img_array.reshape(-1,1)
    kmeans = KMeans(n_clusters = 2, random_state = 99).fit(img_reshaped)
    kmeans_y = kmeans.fit_predict(img_reshaped)
    lab = kmeans.labels_
    mask = lab.reshape(img_array.shape)
    # Control the mask
    if mask[1,1] == 0:
        mask = abs(mask-1)
    return mask

# Function to make the mask more robust
def fill_mask(first_mask):
    cleared = clear_border(first_mask)
    selem = disk(1.5)
    binary = binary_erosion(cleared, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    final_mask = binary.astype('uint8')
    # Control the mask
    n_0 = len(final_mask[final_mask==0])
    n_1 = len(final_mask[final_mask==1])
    if (n_1/(n_0 + n_1)) < 0.1 or (n_1/(n_0 + n_1)) > 0.9:
        print("The selected scan is not good enough")
    return final_mask

# Function to manually select the lungh mask or any other ROI
def manually_roi_select(lista_img, mask_list, n_mask):
    fig = plt.figure()
    plt.imshow(lista_img[int(n_mask)-1], interpolation='nearest', cmap=plt.cm.gray)
    plt.colorbar()
    plt.title("left click: line segment right click or double click: close region")
    roi1 = RoiPoly(color='r', fig=fig)
    mask = roi1.get_mask(mask_list[int(n_mask)-1])
    mask_list[int(n_mask)-1] = mask
    return roi1, mask

# Function to show the manually selected ROI
def show_roi_selected(roi1, mask, n_mask, lista_img):
    fig, ([ax1, ax2]) = plt.subplots(1,2, figsize=(15,12))
    roi1.display_roi()
    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.set_title("Maschera")
    ax2.imshow(lista_img[int(n_mask)-1], cmap=plt.cm.gray)
    ax2.set_title("Maschera sull'immagine originale")
    return None

# This function take the image and extract the Haralick Feature matrix (4x13)
def extract_haralick(img):
    # adding gaussian filter
    img_test = mahotas.gaussian_filter(img, .5, cval = 100, mode = "reflect")
    # setting threshold (threshed is the mask)
    threshed = (img_test > np.quantile(img_test[img_test!=0],.95))
    # making is labeled image
    feature_map, n = mahotas.label(threshed)
    # getting haralick features
    h_feature = mahotas.features.haralick(feature_map, distance = 2)    
    return feature_map, h_feature

# Function for extract and show the equalized and filtered image (dst) and the Haralick map
# for the three scans selected for a patient
def features_and_plot(lista_img, lista_mask, printing=False):
    
    id_feat_list = []
    row, col = lista_img[0].shape
    
    for cont in range(0,len(lista_img)):
        # Equalized hist of image
        dst = cv2.equalizeHist(lista_img[cont]*lista_mask[cont])
        # Wiener filter
        filtered_img = wiener(dst, (3, 3), noise=10)
        # Extract Haralick Features
        feature_map, h_feature = extract_haralick(filtered_img)
        id_feat_list.append(h_feature)

        # showing the features
        if printing:
            fig, ([ax1, ax2]) = plt.subplots(1,2, figsize=(12,10))
            fig.suptitle("Scans: " + str(cont + 1), y = 0.8, fontsize=16)
            ax1.imshow(dst, cmap=plt.cm.gray)
            ax1.set_title("Equalized image (Masked)")
            ax2.imshow(feature_map, cmap=plt.cm.viridis)
            ax2.set_title("Haralick Features Map")
            plt.tight_layout()         
    return id_feat_list   

# Function to predict the probability that the patient is positive or negative to Covid-19
def covid_predict(feat):
    lista_pat=[]
    for elem in feat:
        lista_pat.append(elem.reshape(-1))
    pat_array = np.vstack(lista_pat)
    pat_array = np.reshape(lista_pat, (156))
    pat_array = pat_array.reshape(1,-1)
    input_model = pca.transform(pat_array)
    idx_predicted = model.predict(input_model)
    proba = model.predict_proba(input_model)[0,idx_predicted]
    proba = np.around(proba,4) * 100
    if idx_predicted == 0:
        print("The patient is negative to Covid-19 with a confidence of " + str(proba[0]) + "%")
    else:
        print("The patient is positive to Covid-19 with a confidence of " + str(proba[0]) + "%")
    return None



