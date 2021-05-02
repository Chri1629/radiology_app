
class Patient:
    def __init__(self, dimx, dimy, dimz, width, height, info):
        self.slices = []
        self.sel_slices = []
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.width = width
        self.height = height
        self.dimvoxel = self.dimx * self.dimy * self.dimz
        self.info = info
        self.positivity = None

    def add_slice(self, s):
        self.slices.append(s)

    def get_sel_slices(self):
        n_slices = len(self.slices)
        if n_slices < 50:
            print("Incomplete Exam, select manually the slices")
        else:
            n_dcm_interval = int(n_slices / 12)
            lista_idx = [n_dcm_interval*4, n_dcm_interval*5, n_dcm_interval*6]
            self.sel_slices = [self.slices[i] for i in lista_idx]

class Slices:

     def __init__(self):
         # slices = Patient.slices
         sel_slices = Patient.sel_slices
         mask = self.get_mask(self.sel_slices)
         masked_slices = self.mask_x_slice(self.mask)
         feature_map, feat = self.get_haralick(self.masked_slices)
         positivity = self.get_positivity(feat)

   
     def get_mask(self):
        mask = []
        for slice in self.sel_slices:
            img_reshaped = slice.reshape(-1,1)
            kmeans = KMeans(n_clusters = 2, random_state = 99).fit(img_reshaped)
            kmeans_y = kmeans.fit_predict(img_reshaped)
            lab = kmeans.labels_
            tmp_mask = lab.reshape(img_array.shape)
            # Control the mask
            if tmp_mask[1,1] == 0:
                tmp_mask = abs(tmp_mask-1)

            # Fill the mask
            cleared = clear_border(tmp_mask)
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
            mask.append(final_mask)
        return mask

     def mask_x_slice(self):
         masked_slices = []
         for i in range(0,len(self.sel_slices)):
             masked_slices.append(self.sel_slices[i] * self.mask[i])
         return masked_slices

     def get_haralick(self):
         feature_map_list = []
         h_feature_list = []
         for mask_slice in self.masked_slices:
              img_test = mahotas.gaussian_filter(mask_slice, .5, cval = 100, mode = "reflect")
              # setting threshold (threshed is the mask)
              threshed = (img_test > np.quantile(img_test[img_test!=0],.95))
              # making is labeled image
              feature_map, n = mahotas.label(threshed)
              feature_map_list.append(feature_map)
              # getting haralick features
              h_feature = mahotas.features.haralick(feature_map, distance = 2)    
              h_feature_list.append(h_feature)
         return feature_map_list, h_feature_list

     def get_positivity(self):
        lista_pat=[]
        for elem in self.feat:
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
            self.positivity = 0
            Patient.positivity = 0
        else:
            print("The patient is positive to Covid-19 with a confidence of " + str(proba[0]) + "%")
            self.positivity = 1
            Patient.positivity = 1

