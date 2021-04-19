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
from classes import Slice, Tac, Lesion
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import cluster, mixture, metrics

def loader(PathDicom):
    tacs = {}
    paths = []
    for sbj in tqdm(os.listdir(f'{PathDicom}/brain_data')):
            if sbj[0] != '.':

                tacs[sbj] = {}

                for data in os.listdir(f'{PathDicom}/brain_data/{sbj}'):
                    if data[0] != '.':
                        tacs[sbj][data] = {}
                        for tac in os.listdir(f'{PathDicom}/brain_data/{sbj}/{data}'):
                            if tac[0] != '.':
                                path = f'{PathDicom}/brain_data/{sbj}/{data}/{tac}'
                                paths.append(path)

                                file_list_sorted_dir = load_file(path, last_4chars)
                                dimx, dimy, dimz, dim_voxel, ConstPixelDims, ConstPixelSpacing, RefDs = extract_info(file_list_sorted_dir[0], len(file_list_sorted_dir))
                                ArrayDicom, stackimg, nvoxel = create_array_dicom(file_list_sorted_dir, ConstPixelDims, RefDs)

                                info = {'SUBJECT': sbj, 'DATE': data, 'SERIES': tac,
                                'SEX': RefDs.PatientSex, 'HEIGHT': float(RefDs.PatientSize), 'WEIGHT': float(RefDs.PatientWeight)}
                                # creazione oggetto tac
                                ct = Tac(dimx, dimy, dimz, ArrayDicom.shape[1], ArrayDicom.shape[2], info)
                                # crea le slice per l'oggetto tac
                                for i in range(ArrayDicom.shape[0]):
                                    ct.add_slice(Slice(stackimg[i,:,:]))


                                # aggiunge alla lista delle nostre tac
                                tacs[sbj][data][tac] = ct
    return tacs


def create_cluster(dati,saving_path):
    col_names = ['surface', 'volume', 'R_equiv', 'sphericity',
                          'disproportion', 'surfacevolume_ratio']

    features = dati[col_names]
    ct = ColumnTransformer([
            ('somename', StandardScaler(), ['surface', 'volume', 'R_equiv', 'sphericity',
                                  'disproportion', 'surfacevolume_ratio'])], remainder='passthrough')

    dati_scaled = dati.copy()
    dati_scaled[col_names] = ct.fit_transform(features)
    dati_scaled.drop(['subjectID'], axis = 1, inplace = True)
    two_means = cluster.MiniBatchKMeans(n_clusters= 2, random_state = 4242)
    two_means.fit(dati_scaled)
    y_pred = two_means.predict(dati_scaled)
    dati_with_cluster = dati.copy()
    dati_with_cluster['cluster'] = y_pred
    dati_with_cluster.to_csv(saving_path+"/cluster.csv",index=False)


class main:
    if __name__ == "__main__":

        print("#"*40)
        print("#            TAC SUPPORT APP           #")
        print("#"*40,"\n")

        #PathDicom = "/Users/christianuccheddu/Desktop/uni/medical_imaging" # fino al path brain data
        PathDicom = "/Users/fede9/Desktop/imageMedicalLab/medical_imaging/"
        #PathDicom = "/Users/pietrobonardi/Downloads/"

        print("LOADING ALL DATA \n...please wait...")
        tacs = loader(PathDicom)
        print("\nThe application is ready:\nIt is now possible to visualize and analyze patient's CT\n\n")
        esc0=True
        if os.path.exists(PathDicom+"/dati.csv"):
            dati = pd.read_csv(PathDicom+"/dati.csv",sep=",")
        else:
            dati = pd.DataFrame(columns = ['subjectID', 'surface', 'volume', 'R_equiv', 'sphericity','disproportion', 'surfacevolume_ratio'])

        while(esc0):

            print("Select a subject")
            for i,k in enumerate(list(tacs.keys())):
                print(f"{i}. {k}")
            sogg = int(input("\n> "))
            selected_sogg = list(tacs.keys())[sogg]
            print("-"*40)
            print(f"The dates {selected_sogg} are:")
            for i,k in enumerate(list(tacs[selected_sogg].keys())):
                print(f"{i}. {k}")
            data = int(input("\n> "))
            selected_data = list(tacs[selected_sogg].keys())[data]

            print("-"*40)
            print(f"Available CT {selected_sogg} in {selected_data} are:")
            for k in list(tacs[selected_sogg][selected_data].keys()):
                print(k)
            ct = input("\n> ")

            selected_tac = tacs[selected_sogg][selected_data][ct]

            print("-"*40)
            print(f"Info {selected_sogg}:\n",selected_tac.info)

            ## DISPLAY
            MedicalPlot.complete_dicom(selected_tac.bed_masked_slices, selected_tac.norm)
            print(f"Do you want to save the tac as gif? Press \"y\", else any buttons")
            risp = input("\n> ")
            if risp.lower() == 'y':
                MedicalPlot.create_gif(selected_tac.bed_masked_slices, selected_tac.norm, 100, f"{PathDicom}/brain_pics/{selected_sogg}/{selected_data}/{ct}")

            ## SLICE SELECTION
            print("-"*40)

            esc1=True
            while(esc1):
                print("\nDo you want to select a specific slice? Press \"y\", else any buttons")
                res = input("\n> ")
                res = res.lower()
                if res == "y":
                    print("Which slice?")
                    slc = int(input("\n> "))
                    MedicalPlot.plot_slice(selected_tac.bed_masked_slices[slc-1],selected_tac.norm,n_slice = slc)
                else:
                    esc1=False

            ## CROPPING
            esc2 = True
            lesion_masks = []
            print("-"*40)
            print("\nSelect a starting slice in order to crop")
            s_current = int(input("\n> "))
            s_start = s_current - 1
            slice_current = selected_tac.bed_masked_slices[s_current-1]

            while (esc2):
                #Let user draw ROI
                mask = MedicalPlot.manual_crop(slice_current, selected_tac.norm,n_slice = s_current)
                print("\n\nLesion mask created")
                lesion_masks.append(mask)
                print("-"*40)
                q = input("Still cropping?\nIf yes press any buttons, else \"q\"\n> ")
                q = q.lower()
                if (q == 'q'):
                    esc2 = False
                s_current += 1
                slice_current = selected_tac.bed_masked_slices[s_current-1]
            MedicalPlot.plot_masks(lesion_masks, s_start+1)

            ## RESULTS
            print("\nExtracting Features\n ... please wait ... \n\n")
            lesion = Lesion(np.stack(lesion_masks), selected_tac.dimx, selected_tac.dimy, selected_tac.dimz)
            print("#"*40)
            print(f"#            {selected_sogg.upper()}              #")
            print("#"*40,"\n")

            print(f"- Volume: {lesion.volume} cm3\n- Surface: {lesion.surface} cm2\n- Sfericity: {lesion.sphericity}\n- Disproportion: {lesion.disproportion}\n- Equivalent radius: {lesion.R_equiv}\n- Surface volume ratio: {lesion.surfacevolume__ratio} cm-1")
            dati = dati.append({'subjectID':selected_tac.info["SUBJECT"],"surface":lesion.surface,'volume':lesion.volume, 'R_equiv':lesion.R_equiv, 'sphericity':lesion.sphericity,'disproportion':lesion.disproportion, 'surfacevolume_ratio':lesion.surfacevolume__ratio},ignore_index=True)
            print(f"\n{dati}")
            print("-"*40)
            q = input("\nDo you want to see another subject?\nIf yes press \"y\" else \"q\"\n> ")
            q = q.lower()
            if (q == 'q'):
                esc0 = False
            print("\n\n\n\n")
            print("#"*40)
            print("#            TAC SUPPORT APP           #")
            print("#"*40,"\n")
        
        dati.to_csv(PathDicom+"/dati.csv",index=False)
        print("Do you want to generate a cluster of your data?\nIf yes press \"y\", else press any buttons")
        cls = input("\n> ")
        cls = cls.upper()
        if cls == "Y":
            create_cluster(dati,PathDicom)
