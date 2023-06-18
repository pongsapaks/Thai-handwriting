#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import os
import glob
import numpy as np
import pickle
import math


def clean_data():
    directory = "../DADS5001 Machine Learning/data/clean"

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

#    dirtys = [
#        '7c9108fe-b240-4632-a024-f1ee922962ec',
#        '20_a2178975-acff-4afe-88b9-f6fee8694ceb',
#        'de366cab-6532-42ed-9926-38351927019b',
#        '76c2e443-c8d1-40b0-96a9-073548c9617b',
#        '21_e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
#        'e95ad3b8-30cb-47ad-9f68-2a5bb7aeb5bb',
#        'fb05cb2a-c27b-4476-8cff-74f5ddbc8224',
#        '078c1b18-e672-466d-a30b-f49a81710be6',
#        '67ce79dc-de9c-4956-ad7b-fabf7aa9c6fa',
#        '729207eb-f3f7-46e2-986a-74f990296da4',
#        '420994cc-5e99-42eb-84b6-2392486a33b6',
#        '0a9af826-aaf4-45da-9d46-e1b5dc486264'
#    ]

    os.system('cp -r ../DADS5001 Machine Learning/data/raw/* ../DADS5001 Machine Learning/data/clean/')
    os.system('mv ../DADS5001 Machine Learning/data/clean/10 ../DADS5001 Machine Learning/data/clean/0')

#    for i in range(0, 10):
#        for dirty in dirtys:
#            path = directory + '/' + str(i) + '/' + dirty + '.png'
#            os.remove(path)

import os
import glob
import requests
from PIL import Image
import numpy as np
import pickle

def make_dataset(data_url="https://github.com/pongsapaks/Thai-handwrittingnumberproject/tree/main/raw", size=28):
    X = []
    Y = []
    
    response = requests.get(data_url)
    if response.status_code == 200:
        content = response.text
        for line in content.split('\n'):
            if 'blob-code blob-code-inner js-file-line' in line:
                image_url = line.split('src="')[1].split('"')[0]
                label = os.path.basename(os.path.dirname(image_url))
                
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    img = Image.open(response.raw).convert('L').resize((size, size))
                    img = ImageOps.invert(img)
                    x = np.array(img)
                    
                    X.append(x)
                    Y.append(label)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    data = {"X": X, "Y": Y}
    pickle.dump(data, open("thainumber_{}.pkl".format(size), "wb"), protocol=2)


def load_dataset(size = 28):
    data = pickle.load(open("thainumber_{}.pkl".format(size), "rb"))
    X = data['X']
    Y = data['Y']
    return X, Y

def prepare_input(file):
	img = load_img(file, grayscale=True, target_size=(28, 28))
	img = ImageOps.invert(img)
	x = img_to_array(img)
	return x

def img_cloud_dataset(size = 28):
    X, Y = load_dataset(size)
    x = 0
    y = 0
    new_im = Image.new('L', (size * 50, size * math.ceil(X.shape[0] / 50)))
    for i in range(0, X.shape[0]):
        if (i != 0 and i % 50 == 0):
            y += size
            x = 0

        im = array_to_img(X[i])
        new_im.paste(im, (x, y))
        x += size
    new_im.save("cloud_dataset_{}.png".format(size))


# In[2]:


make_dataset()


# In[3]:


X,Y = load_dataset()


# In[4]:


X


# In[5]:


Y


# In[6]:


import matplotlib.pyplot as plt
plt.imshow(X[0], cmap='gray', vmin=0, vmax=255)


# In[7]:


#Reshape X to change array to dataframe
reshaped_X = X.reshape((X.shape[0], -1))
reshaped_X.shape


# In[8]:


#Change array to dataframe

import pandas as pd
Ydf = pd.DataFrame(Y)
Xdf = pd.DataFrame(reshaped_X)


# In[9]:


Xdf


# In[10]:


Ydf


# In[11]:


#Pycaret

from pycaret.classification import *
clf = setup(Xdf, target = Ydf, train_size = 0.8,
            numeric_imputation = 'median',
            categorical_imputation = 'mode')


# In[12]:


top5_model = compare_models(sort = 'Accuracy', fold = 5, n_select = 5)


# In[13]:


#Scree Plot

from sklearn.decomposition import PCA
pca = PCA(n_components = Xdf.shape[1]) #Maximum component is all features
pca.fit_transform(Xdf) #Fitting PCA
explain_ratio = pca.explained_variance_ratio_
explain_ratio_cum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize = (15, 100))
ax = plt.axes()
ax.set_facecolor('#dfe3e6')
plt.grid(color = 'w')
plt.xlabel('Number of components')
plt.ylabel('variance explained')
plt.title('Scree plot')

plt.plot(range(1, explain_ratio.shape[0] + 1), explain_ratio, c = 'royalblue', marker = 'o', linewidth = 2.5, label = 'Individual')
plt.plot(range(1, explain_ratio.shape[0] + 1), explain_ratio_cum, c = 'firebrick', marker = 'o', linestyle = '--', label = 'Cumulative')

for x, ex_ratio, ex_ratio_cum in zip(range(1, explain_ratio.shape[0] + 1),
                                     explain_ratio,
                                     explain_ratio_cum):
    ex_ratio_label = f'{ex_ratio * 100:.2f}%'
    plt.annotate(ex_ratio_label, (x, ex_ratio), textcoords = 'offset points',
               xytext = (5, 5), ha = 'center')
    ex_ratio_cum_label = f'{ex_ratio_cum * 100:.2f}%'
    plt.annotate(ex_ratio_cum_label, (x, ex_ratio_cum), textcoords = 'offset points',
               xytext = (5, 5), ha = 'center')

plt.show()


# In[14]:


#Standardization
X_mean = Xdf.mean()
X_std = Xdf.std()
Z = (Xdf-X_mean)/X_std
Z = Z.fillna(0)
Z


# In[15]:


#Covariance
c = Z.cov()
c


# In[16]:


#Eiganvalues & Eiganvectors
eiganvalues, eiganvectors = np.linalg.eig(c)
print('Eigan Values:\n', eiganvalues)
print('Eigan Values Shape:', eiganvalues.shape)
print('Eigan Vectors Shape:', eiganvectors.shape)


# In[17]:


#Explained Variance
idx = eiganvalues.argsort()[::-1]
eiganvalues = eiganvalues[idx]
eiganvectors = eiganvectors[:,idx]

explained_var = np.cumsum(eiganvalues)/np.sum(eiganvalues)
explained_var


# In[18]:


#Find n for 80% Explained Variance
n_components = np.argmax(explained_var >= 0.80) +1
n_components


# In[19]:


#Dataframe to Array
Xarray = Xdf.values
Xinverse = Xarray.reshape(-1, 28, 28)
plt.imshow(Xinverse[0], cmap='gray', vmin=0, vmax=255)


# In[20]:


#Apply PCA
pca = PCA(n_components = 165)
pca.fit(Z)
X_pca = pca.transform(Z)
pca_df = pd.DataFrame(X_pca)
pca_df


# In[26]:


clf = setup(pca_df, target = Ydf)


# In[24]:


pca_df


# In[25]:


Ydf


# In[ ]:




