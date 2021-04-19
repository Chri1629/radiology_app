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

class Slice:
    def __init__(self,  img):
        self.img = img

    def get_resolution(self):
        return self.img.shape[0], self.img.shape[1]

class Tac:
    def __init__(self, dimx, dimy, dimz, width, height, info):
        self.slices = []
        self.bed_masked_slices = []
        self.norm = None
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.width = width
        self.height = height
        self.dimvoxel = self.dimx * self.dimy * self.dimz
        self.info = info
        self.lesion = None

    def add_slice(self, s):
        self.slices.append(s)
        self.bed_masked_slices.append(s.img)
        self.norm = self.add_norm()

    def add_norm(self):
        #norm = mpl.colors.Normalize(vmin = np.amin(self.bed_masked_slices), vmax = np.amax(self.bed_masked_slices))
        #norm = mpl.colors.Normalize(vmin = -200, vmax = 150)
        norm = mpl.colors.Normalize(vmin = -10, vmax = 90)
        return norm



class Lesion:
    def __init__(self, focus_lesion, dimx, dimy, dimz):
        self.focus_lesion = focus_lesion
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.dimvoxel = self.dimx * self.dimy * self.dimz
        self.volume = self.get_volume()
        self.surface = self.get_surface()
        self.R_equiv = self.get_req()
        self.sphericity = self.get_sphericity()
        self.disproportion = self.get_sphericaldisp()
        self.surfacevolume__ratio = self.get_survol_ratio()

    def get_volume(self):
        n_voxel_on = self.focus_lesion[self.focus_lesion!=0].shape[0]
        volume = n_voxel_on * self.dimvoxel
        return round(volume/10**3,2)

    def get_surface(self):
        x__size = self.focus_lesion.shape[0]-1 # Size in terms of # voxels
        y__size = self.focus_lesion.shape[1]-1 # Size in terms of # voxels
        z__size = self.focus_lesion.shape[2]-1 # Size in terms of # voxels

        dx = self.dimz/10
        dy = self.dimx/10
        dz = self.dimy/10

        surface = 0
        N = np.zeros((self.focus_lesion.shape[0], self.focus_lesion.shape[1],self.focus_lesion.shape[2]))
        for k in range(z__size):
            for j in range(y__size):
                if self.focus_lesion[0,j,k] > 0:
                    surface = surface+(dz*dy)
                    N[0,j,k] = 1
                if self.focus_lesion[x__size,j,k] > 0:
                    surface = surface+(dz*dy)
                    N[x__size,j,k] = 1

        for k in range(z__size):
            for i in range(x__size):
                if self.focus_lesion[i,0,k] > 0:
                    surface = surface+(dz*dx)
                    N[i,0,k] = 1
                if self.focus_lesion[i,y__size,k] > 0:
                    surface = surface+(dz*dx)
                    N[i,y__size,k] = 1

        for j in range(y__size):
            for i in range(x__size):
                if self.focus_lesion[i,j,0] > 0:
                    surface = surface+(dy*dx)
                    N[i,j,0] = 1
                if self.focus_lesion[i,j,z__size] > 0:
                    surface = surface+(dy*dx)
                    N[i,j,z__size] = 1

        for i in range(x__size):
            for j in range(y__size):
                for k in range(z__size):
                    if k > 0 and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i,j,k-1] == 0:
                        surface = surface+(dx*dy)
                        N[i,j,k] = 1
                    if k < z__size and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i,j,k+1] == 0:
                        surface = surface+(dx*dy)
                        N[i,j,k] = 1
                    if i > 0 and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i-1,j,k] == 0:
                        surface = surface+(dz*dy)
                        N[i,j,k] = 1
                    if i < x__size and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i+1,j,k] == 0:
                        surface = surface+(dz*dy)
                        N[i,j,k] = 1
                    if j > 0 and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i,j-1,k] == 0:
                        surface = surface+(dz*dx)
                        N[i,j,k] = 1
                    if j < y__size and self.focus_lesion[i,j,k] > 0 and self.focus_lesion[i,j+1,k] == 0:
                        surface = surface+(dz*dx)
                        N[i,j,k] = 1

        return round(surface,2)

    def get_req(self):
        return round((self.volume*3/(4*np.pi))**(1/3),2)

    def get_sphericaldisp(self):
        return round(self.surface/(4*np.pi*(self.R_equiv)**2),2)

    def get_sphericity(self):
        return round(((np.pi**(1/3))*((6*self.volume)**(2/3)))/self.surface,2)

    def get_survol_ratio(self):
        return round(self.surface/self.volume,2)
