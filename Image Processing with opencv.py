#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import glob,cv2
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[12]:


import os
os.chdir('C:\\Analytics\\Deep Learning\\Image processing\\image')


# In[13]:


import matplotlib.image as mpimg
img = mpimg.imread('seafishing.jpg')
imgplot = plt.imshow(img)
plt.show()


# In[14]:


def rotate(image,angle,center=None,scale=1.0):
    (h,w) = img.shape[:2]
    if center is None:
        center = (w/2,h/2)
        M = cv2.getRotationMatrix2D(center,angle,scale)
        rotated = cv2.warpAffine(img,M,(w,h))
    return rotated


# In[15]:


image_rotate = rotate(img,60)
plt.imshow(image_rotate)
plt.show()


# Translate the image

# In[16]:


def translate(image,x,y):
    M = np.float32([[1,0,x],[0,1,y]])
    shifted = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return shifted


# In[17]:


img_translate = translate(img,200,100)
plt.imshow(img_translate)
plt.show()


# Flip Horizontal

# In[18]:


img_flip = cv2.flip(img,1)
plt.imshow(img_flip)
plt.show()


# Filp vertical

# In[19]:


img_flip = cv2.flip(img,0)
plt.imshow(img_flip)
plt.show()


# Day image to night

# In[23]:


def day_to_night(image):
    arr = image*np.array([0.1,0.54,1.7])
    img = (255*arr/arr.max()).astype(np.uint8)
    return img


# In[24]:


img_d2n = day_to_night(img)
plt.imshow(img_d2n)
plt.show()


# Shear Image - called Affline Transformation

# In[25]:


def shear_image(image,shear=0.2):
    from skimage import transform
    afline = transform.AffineTransform(shear=shear)
    modified = transform.warp(image,afline)
    return modified


# In[26]:


img_shear = shear_image(img,shear=0.5)
plt.imshow(img_shear)
plt.show()


# Add light to nigh images

# In[29]:


def bright_image(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h+=60
    final_hsv = cv2.merge((h,s,v))
    image = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2RGB)
    return image


# In[30]:


bright_img = bright_image(img_d2n)
plt.imshow(bright_img)
plt.show()


# In[ ]:




