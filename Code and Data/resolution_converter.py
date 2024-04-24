import numpy as np
import cv2
import matplotlib.pyplot as plt

load_dir = '/Users/Austin Kwon/Desktop/Price Checker V2/item_images_300x300/'
save_dir = '/Users/Austin Kwon/Desktop/Price Checker V2/item_images_120x120/'
item_name = 'Vitamin_Code_Grow_Bone'

Xtrain = np.load(load_dir + item_name + '.npz')['X']
Xtest = np.load(load_dir + item_name + '_Test.npz')['X']

Xtrain0 = np.zeros((len(Xtrain), 120, 120, 3))
Xtest0 = np.zeros((len(Xtest), 120,120,3))

for i in range(len(Xtrain)):
    if i < len(Xtest):
        Xtest0[i] = cv2.resize(Xtest[i], (120,120))/255
    Xtrain0[i] = cv2.resize(Xtrain[i], (120,120))/255
    
Xtrain0 = Xtrain0.astype(np.float16)
Xtest0 = Xtest0.astype(np.float16)

plt.imshow(np.hstack(( (Xtrain0[np.random.randint(len(Xtrain))]).astype(np.float64), 
                      (Xtest0[np.random.randint(len(Xtest))]).astype(np.float64) )))

np.savez(save_dir + item_name + '_120.npz', X = Xtrain0)
np.savez(save_dir + item_name + '_120_Test.npz', X = Xtest0)