
# Commented out IPython magic to ensure Python compatibility.
import time
start_time1 = time.time()
!pip install spectral
import spectral
from google.colab import drive, files
drive.mount('/content/drive')
## Basics
import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore')
import numpy
import numpy as np
import pandas as pd
from PIL import Image
from operator import truediv
import scipy.io as sio
import os
import seaborn as sns
## Ploting
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
# %matplotlib inline
## Sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
## Dimensionality Reduction Methods
from sklearn.decomposition import PCA
## Deep Model
import keras, h5py
from keras.layers import Input, Conv2D, Conv3D, Flatten, Dense, Reshape, Dropout
#from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import spectral.io.envi as envi
## Mounting Google Drive
path='/content/drive/MyDrive/HyperBlood/'

## Get HSI Data and Ground Truths
## Get HSI Data
def get_data(name,remove_bands=True,clean=True, path=path):
    """
    Input: name: name; remove_bands: if True, noisy bands are removed (leaving 113 bands)
    clean: if True, remove damaged line
    Output: data, wavelenghts as numpy arrays (float32)
    """
    name = convert_name(name)
    filename = "{}data/{}".format(path,name)
    hsimage = envi.open('{}.hdr'.format(filename),'{}.float'.format(filename))
    wavs = np.asarray(hsimage.bands.centers)
    data = np.asarray(hsimage[:,:,:],dtype=np.float32)
    #removal of damaged sensor line
    if clean and name!='F_2k':
        data = np.delete(data,445,0)
    if not remove_bands:
        return data,wavs
    return data[:,:,get_good_indices(name)],wavs[get_good_indices(name)]

## Get Ground Truths
def get_anno(name,remove_uncertain_blood=True,clean=True, path=path):

    name = convert_name(name)
    filename = "{}anno/{}".format(path,name)
    anno = np.load(filename+'.npz')['gt']
    #removal of damaged sensor line
    if clean and name!='F_2k':
        anno = np.delete(anno,445,0)
    #remove uncertain blood + technical classes
    if remove_uncertain_blood:
        anno[anno>7]=0
    else:
        anno[anno>8]=0
    return anno


def blood_loader(path, name):

    img = np.asarray(get_data(name, path=path)[0], dtype='float32')
    gt = get_anno(name, path=path).astype('uint8')
    gt = np.where(gt == 4, 0, gt)
    for element in [4, 7, 9]:
        gt = np.where(gt == element, element - 1, gt)
    label_values = ["unclassified",
                    "blood",
                    "ketchup",
                    "artificial blood",
                    "poster paint",
                    "tomato concentrate",
                    "acrylic paint"]
    rgb_bands, ignored_labels = (41, 23, 15), [0]
    return img, gt, rgb_bands, ignored_labels, label_values

def convert_name(name):
    name = name.replace('(','_')
    name = name.replace(')','')
    return name

def get_good_indices(name=None):
    name = convert_name(name)
    if name!='F_2k':
        indices = np.arange(123)
        indices = indices[5:-6]
    else:
        indices = np.arange(13)
    indices=np.delete(indices,[43,44,45])
    return indices

## Load HSI Dataset
def LoadHSIData(method):
    ## List of HSI Dataasets
    HSI_list = ['A_1.hdr', 'B_1.hdr', 'C_1.hdr', 'D_1.hdr', 'E_1.hdr', 'E_7.hdr',
                'E_21.hdr', 'F_1.hdr.hdr', 'F_1a.hdr', 'F_1s.hdr', 'F_2.hdr',
                'F_2k.hdr', 'F_7.hdr', 'F_21.hdr']
    for i in range(len(HSI_list)):
        file_name = HSI_list[i].split('.')[0]
        data_path = os.path.join(os.getcwd(),'/content/drive/MyDrive/HyperBlood/data'+str(file_name))
        # print(data_path)
        if os.path.exists(data_path) == False:
            print(str(file_name)+" is Successfully Found")
        else:
            print(str(file_name) + " already exist")
    data_path = os.path.join(os.getcwd(),'/content/drive/MyDrive/HyperBlood/data')
    data_path1 = os.path.join(os.getcwd(),'/content/drive/MyDrive/HyperBlood/anno')
    if method == 'A_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'B_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'C_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'D_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'E_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'E_7':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'E_21':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_1':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_1a':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_1s':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_2':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_2k':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_7':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    elif method == 'F_21':
        HSI = np.asarray(get_data(method, data_path)[0], dtype='float32')
        GT = get_anno(method, data_path1).astype('uint8')
    return HSI, GT

## Dimension Reduction
def DLMethod(method, HSI, NC = 75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == 'PCA':
        pca = PCA(n_components=NC, random_state=2019)
        RHSI = pca.fit_transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI

## Padding and Spatial Patchs
def ZeroPad(HSI, margin = 2):
    NHSI = np.zeros((HSI.shape[0] + 2 * margin, HSI.shape[1] + 2* margin, HSI.shape[2]))
    x_offset = margin
    y_offset = margin
    NHSI[x_offset:HSI.shape[0] + x_offset, y_offset:HSI.shape[1] + y_offset, :] = HSI
    return NHSI

## Compute the Patch to Prepare for Ground Truths
def Patch(HSI,height_index,width_index):
    height_slice = slice(height_index, height_index+WS)
    width_slice = slice(width_index, width_index+WS)
    patch = HSI[height_slice, width_slice, :]
    return patch

def ImageCubes(HSI, GT, WS = 5, removeZeroLabels = True):
    margin = int((WS - 1) / 2)
    zeroPaddedX = ZeroPad(HSI, margin = margin)
    # split patches
    patchesData = np.zeros((HSI.shape[0] * HSI.shape[1], WS, WS, HSI.shape[2]))
    patchesLabels = np.zeros((HSI.shape[0] * HSI.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = GT[r-margin, c-margin]
            patchIndex = patchIndex + 1
    return patchesData, patchesLabels

## Classification Reports
def ClassificationReports(TeC, Te_Pred):
    Te_Pred = np.argmax(Te_Pred, axis=1)
    target_names = ["blood",
                    "ketchup",
                    "artificial blood",
                    "poster paint",
                    "tomato concentrate",
                    "acrylic paint"]
    classification = classification_report(np.argmax(TeC, axis=1), Te_Pred, target_names = target_names)
    oa = accuracy_score(np.argmax(TeC, axis=1), Te_Pred)
    confusions = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred)
    counter = confusion.shape[2]
    list_diag = np.diag(confusions)
    list_raw_sum = np.add(confusions, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    aa = np.median(each_acc)
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1), Te_Pred)
    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100, target_names

## Writing Classification Results
def CSVResults(file_name, classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa):
    classification = str(classification)
    confusion = str(confusions)
    with open(file_name, 'w') as CSV_file:
      CSV_file.write('{} Tr_Time'.format(Tr_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Te_Time'.format(Te_Time))
      CSV_file.write('\n')
      CSV_file.write('{} DL_Time'.format(DL_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Kappa accuracy (%)'.format(kappa))
      CSV_file.write('\n')
      CSV_file.write('{} Overall accuracy (%)'.format(oa))
      CSV_file.write('\n')
      CSV_file.write('{} Average accuracy (%)'.format(aa))
      CSV_file.write('\n')
      CSV_file.write('\n')
      CSV_file.write('{}'.format(classification))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(confusion))
    return CSV_file

## Plot Ground Truths
def GT_Plot(RDHSI, GT, model, WS):
    height, width = np.shape(GT)
    outputs = np.zeros((height, width))
    for AA in range(height):
      for BB in range(width):
        target = int(GT[AA,BB])
        if target == 0:
          continue
        else :
          image_patch = Patch(RDHSI,AA,BB)
          X_test_image = image_patch.reshape(1,image_patch.shape[0],
                                             image_patch.shape[1],
                                             image_patch.shape[2],
                                             1).astype('float32')
          prediction = (model.predict(X_test_image))
          prediction = np.argmax(prediction, axis=1)
          outputs[AA][BB] = prediction+1
    return outputs

## Plot and Save Confusion Matrix
def Conf_Mat(Te_Pred, TeC, target_names):
    plt.rcParams.update({'font.size': 12})
    Te_Pred = np.argmax(Te_Pred, axis=1)
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred, labels=np.unique(np.argmax(TeC, axis=1)))
    cm_sum = np.sum(confusion, axis=1, keepdims=True)
    cm_perc = confusion / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion).astype(str)
    nrows, ncols = confusion.shape
    for l in range(nrows):
      for m in range(ncols):
        c = confusion[l, m]
        p = cm_perc[l, m]
        if l == m:
          s = cm_sum[l]
          annot[l, m] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
          annot[l, m] = ''
        else:
          annot[l, m] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(confusion, index=np.unique(target_names), columns=np.unique(target_names))
    return cm, annot

## Global Parameters
HSID = "E_1"    ## 'A_1.hdr', 'B_1.hdr', 'C_1.hdr', 'D_1.hdr', 'E_1.hdr', 'E_7.hdr',
                ## 'E_21.hdr', 'F_1.hdr.hdr', 'F_1a.hdr', 'F_1s.hdr', 'F_2.hdr',
                ## 'F_2k.hdr', 'F_7.hdr', 'F_21.hdr'
DLM = "PCA"     ## "PCA", "iPCA", "SPCA", "KPCA", "SVD"
WS = 9          ##(WS-window size) 9, 11, 13, 15, 17, 19, 21, 23, 25
TeRatio = 0.90  ## Percentage of Test Samples
VeRatio = 0.50  ## Percentage of Validation Samples
adam = Adam(learning_rate = 0.001, decay = 1e-04)
adam= tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay = 1e-04)

HSI, GT = LoadHSIData(HSID)
k = 10
print("# of Dimensions", k)          ## Dimensions
print("Shape of HSI", HSI.shape)
print("Shape of GT", GT.shape)
# UC = np.unique(label_values)
UC = np.unique(GT)
print("Name of Classes", UC)
Num_Classes = len(UC)
print("# of Classes", Num_Classes)

## Reduce the Dimensionality
start = time.time()
end = time.time()
DL_Time = end - start
print("The Reduce the Dimensionality took", DL_Time, "seconds to run.")
print("The Reduce the Dimensionality took", DL_Time/60, "minutes to run.")

## Create Image Cubes for Model Building
CRDHSI, CGT = ImageCubes(RDHSI, GT, WS = WS)
print("Shape of HSI", CRDHSI.shape)
print("Shape of GT", CGT.shape)

## Split Train and Test sets (for 2D and 3D Models)
Tr, Te, TrC, TeC = TrTeSplit(CRDHSI, CGT, TeRatio)
## Split Train and Validation (for 2D and 3D Models)
Tr, Va, TrC, VaC = TrTeSplit(Tr, TrC, VeRatio)
## Reshape Train, Validation, and Test sets
Tr = Tr.reshape(-1, WS, WS, k, 1)
print('Tr', Tr.shape)
TrC = np_utils.to_categorical(TrC)
print('TrC', TrC.shape)
Va = Va.reshape(-1, WS, WS, k, 1)
print('Va', Va.shape)
VaC = np_utils.to_categorical(VaC)
print('VaC', VaC.shape)
Te = Te.reshape(-1, WS, WS, k, 1)
print('Te', Te.shape)
TeC = np_utils.to_categorical(TeC)
print('TeC', TeC.shape)

## A Fast and Compact 3D CNN Model

start_timeF3D = time.time() ##Timer Count starts
input_layer = Input((WS, WS, k, 1))
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
output_layer = Dense(units = Num_Classes, activation='softmax')(dense_layer2)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
## Training Time and Fit the Model
start_Tr = time.time()
history1 = model.fit(x=Tr, y=TrC, batch_size=256, epochs=10, validation_data=(Va, VaC))
end_Tr = time.time()
Tr_Time = end_Tr - start_Tr
Tr_Time_F3D = end_Tr - start_Tr
## Prediction Model
startTe = time.time()
Te_Pred = model.predict(Te)
endTe = time.time()
Te_Time = endTe - startTe
Te_Time_Te = endTe - startTe
## Prediction and Computing the Accuacy
classification_f3d,confusion,oa,each_acc,aa,kappa,target_names = ClassificationReports(TeC, Te_Pred)
print(classification_f3d)


end_timeF3D = time.time()##Timer Ends
run_timeF3D = end_timeF3D - start_timeF3D
print("Training Time and Fit the Model  : ", Tr_Time_F3D, "seconds to run.")
print("Prediction Model                 : ", Te_Time_Te, " seconds to run.")
print("A Fast and Compact 3D CNN Model  : ", run_timeF3D, "   seconds to run.")

## Writing Confusion Matrix
start_timef3m = time.time() ##Timer Count starts
file_name = str(WS)+str(DLM)+str(k)+"_Confusion_Matrix_Fast_3D_CNN_with_biggest_font.png"
cm, annot = Conf_Mat(Te_Pred, TeC, target_names)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted for Fast 3D CNN'
fig, ax = plt.subplots(figsize=(14,14))
sns.set(font_scale=1.2)
sns.heatmap(cm, cmap= "Spectral", annot=annot, fmt='', ax=ax, linewidths=0.6, annot_kws={"size": 18})
plt.savefig(file_name, dpi=500)
#files.download(file_name)
end_timef3m = time.time()##Timer Ends
run_timef3m = end_timef3m - start_timef3m
print("The code took", run_timef3m, "seconds to run.")

## Hybrid 3D/2D CNN Model

start_timeHy = time.time() ##Timer Count starts
input_layer = Input((WS, WS, k, 1))
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
##conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 1), activation='relu')(conv_layer3)
## Conv3D with kernel_size=(3, 3, 1) can be used as well instead to reshape and Con2D.
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
dense_layer1 = Dense(units=256, activation = 'relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units = 128, activation = 'relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units = Num_Classes, activation = 'softmax')(dense_layer2)
# Define the Model with Input and Output Layers
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()
## Compiling Hybrid CNN
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
## Fit the Model
start_Tr = time.time()
history2 = model.fit(x = Tr, y = TrC, batch_size = 256, epochs = 10, validation_data = (Va, VaC))
end_Tr = time.time()
Tr_Time = end_Tr - start_Tr
Tr_Time_Hy = end_Tr - start_Tr
## Prediction Model
start_Te = time.time()
Te_Pred = model.predict(Te)
end_Te = time.time()
Te_Time = end_Te - start_Te
Te_Time_Hy = end_Te - start_Te
## Prediction and Computing the Accuacy
classification_hy,confusion,oa,each_acc,aa,kappa,target_names = ClassificationReports(TeC, Te_Pred)
print(classification_hy)

end_timeHy = time.time()##Timer Ends
run_timeHy = end_timeHy - start_timeHy
print("Traing Time & Fit the Model:", Tr_Time_Hy, "seconds to run.")
print("Prediction Model           :", Te_Time_Hy, "seconds to run.")
print("Hybrid 3D/2D CNN Model     :", run_timeHy, "seconds to run.")

## Loss and Accuracy
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title(str(HSID)+'_Loss')
plt.legend(['Tr-3-D','Val-3-D', 'Tr-Hybrid','Val-Hybrid'], loc='upper right')
plt.savefig(str(HSID)+"_loss_curve.png")
plt.show()
#files.download(str(HSID)+"_loss_curve.png")

plt.figure(figsize=(7,7))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title(str(HSID)+'_Accuracy')
plt.legend(['Tr-3-D','Val-3-D', 'Tr-Hybrid','Val-Hybrid'], loc='lower right')
plt.savefig(str(HSID)+"_acc_curve.png")
plt.show()
#files.download(str(HSID)+"_acc_curve.png")
end_time1 = time.time()

print("The Whole code took", end_time1 - start_time1, "seconds to execute")
run_time1 = end_time1 - start_time1
print("The Whole code took", run_time1/60, "minutes to run.")
