import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model

from sklearn.linear_model import Lasso

import cv2
import time

root_directory = '/Users/Austin Kwon/Desktop/Price Checker V2/'

excel_file_name = 'item_log.xlsx'
excel_file_directory = root_directory + excel_file_name
item_names = (pd.read_excel(excel_file_directory).values)[:,1]
item_store_names = (pd.read_excel(excel_file_directory).values)[:,2]
item_prices = (pd.read_excel(excel_file_directory).values)[:,3]
rtable = (pd.read_excel(excel_file_directory).values)[:,4:12]

model_name = 'M50_87480.h5'
model0 = load_model(root_directory + 'trainer and tester/' + model_name)
desired_layer_output = model0.get_layer('flatten').output
conv0 = Model(inputs=model0.input, outputs=desired_layer_output)

model_name = 'M50_86360.h5'
model1 = load_model(root_directory + 'trainer and tester/' + model_name)
desired_layer_output = model1.get_layer('flatten').output
conv1 = Model(inputs=model1.input, outputs=desired_layer_output)

Xtrain0 = np.load(root_directory + 'trainer and tester/Xtrain_ls_0.npz')['Xtrain_ls']
Xtrain1 = np.load(root_directory + 'trainer and tester/Xtrain_ls_1.npz')['Xtrain_ls']

img_path = root_directory + 'item_price_images/'
start_time = time.time()

def resize_and_position_window(window_name, height, width,x_position, y_position):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, x_position, y_position)
    
resize_and_position_window('Webcam 1', 650, 1300, 0,0)
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
c = 0
imgHeight = 140
imgWidth = 105
prediction = 10000000
background = 230/255*np.ones((480, 650, 3))
background[10:400, 10:350, :] = 230/255
black_center = [390/2+10, 640+10+340/2]
black_dims = [330, 380]
lowerground = 230/255*np.ones((170, 640 + 650, 3))
instruction_text = 'Please place the item (must be face up) in lightbox'
disclaimer_text = 'Disclaimer: for precise dietary information, please contact manufacturer'

pred_label = 0
msize = 25
img_no = cv2.imread(img_path + 'no.png')
img_no = np.expand_dims(cv2.resize(img_no, (msize,msize))/255, axis = 0)
img_yes = cv2.imread(img_path + 'yes.png')
img_yes = np.expand_dims(cv2.resize(img_yes, (msize,msize))/255, axis = 0)
img_yn = np.vstack((img_no, img_yes))

if prediction == 10000000:
    img2 = cv2.imread(img_path + 'blank.png')
    if np.shape(img2)[0]/black_dims[1] >= np.shape(img2)[1]/black_dims[0]:
        img_width = int(np.shape(img2)[1]/(np.shape(img2)[0]/black_dims[1]))
        img_height = int(np.shape(img2)[0]/(np.shape(img2)[0]/black_dims[1]))
    else:
        img_width = int(np.shape(img2)[1]/(np.shape(img2)[1]/black_dims[0]))
        img_height = int(np.shape(img2)[0]/(np.shape(img2)[1]/black_dims[0]))
    img2 = cv2.resize(img2, (img_width,img_height))

while True:    
    ret0, frame0 = cap0.read()
    frame1 = frame0
    
    if ret0:
        key = cv2.waitKey(1)
        
        if key == 27: # esc
            cap0.release()
            cv2.destroyAllWindows()
            break
        
        if key == 32: # space bar
            img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            img0 = cv2.resize(img0[:, 320-240:320+240,:], (120,120))
            img0 = np.expand_dims(img0, axis = 0)
            
            img_ls0 = conv0.predict(img0/255)
            lasso0 = Lasso(alpha = 0.07, max_iter = 10000)
            lasso0.fit(Xtrain0.T, img_ls0.T)
            coef0 = lasso0.coef_
            coef0 = coef0.reshape((50, 408))
            lassopred0 = np.sum(coef0, axis=1)/np.sum(coef0)
            
            img_ls1 = conv1.predict(img0/255)
            lasso1 = Lasso(alpha=0.07, max_iter = 10000)
            lasso1.fit(Xtrain1.T, img_ls1.T)
            coef1 = lasso1.coef_
            coef1 = coef1.reshape((50, 408))
            lassopred1 = np.sum(coef1, axis=1)/np.sum(coef1)
            
            FCNNpred0 = (model0.predict(img0/255)).flatten()
            FCNNpred1 = (model1.predict(img0/255)).flatten()
            
            prediction = FCNNpred0 + FCNNpred1 + lassopred0 + lassopred1
            
            img2 = cv2.imread(img_path + item_names[np.argmax(prediction.flatten())] + '.png')
            if np.shape(img2)[0]/black_dims[1] >= np.shape(img2)[1]/black_dims[0]:
                img_width = int(np.shape(img2)[1]/(np.shape(img2)[0]/black_dims[1]))
                img_height = int(np.shape(img2)[0]/(np.shape(img2)[0]/black_dims[1]))
            else:
                img_width = int(np.shape(img2)[1]/(np.shape(img2)[1]/black_dims[0]))
                img_height = int(np.shape(img2)[0]/(np.shape(img2)[1]/black_dims[0]))
            img2 = cv2.resize(img2, (img_width,img_height))
            
            if c == 0:
                Xhat = img0
            else:
                Xhat = np.vstack((Xhat, img0))
            c += 1
        
        if c > 0:
            pred_label = np.argmax(prediction.flatten())
            pred_item = item_store_names[pred_label]
            price_text = '$' + '{:.2f}'.format(float(str(item_prices[pred_label])))
        else:
            price_text = '$0.00'
            pred_item = 'Press Space Bar'
            
        item_name_text = pred_item
        frame2 = np.vstack((np.hstack((frame1/255, background)), lowerground))
        frame2[205-int(np.round(img_height/2)):205+int(img_height-np.round(img_height/2)), 
                825-int(np.round(img_width/2)):825+int(img_width-np.round(img_width/2)),:] = img2/255
        if len(item_name_text) < 32:
            cv2.putText(frame2, item_name_text, (650, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        else:
            space_idx = np.array(np.where(np.array(list(item_name_text)) == ' ')).flatten()
            next_line_idx = space_idx[np.argmin(abs(space_idx - 29))]
            cv2.putText(frame2, item_name_text[:next_line_idx], (650, 430), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame2, item_name_text[next_line_idx+1:], (650, 470), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.putText(frame2, price_text, (1000, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 1), 3)
        cv2.putText(frame2, instruction_text, (5, 505), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        cv2.putText(frame2, disclaimer_text, (5, 505+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (1, 0, 0), 2)
        
        if c > 0:
            restrictions = rtable[pred_label]
            confidence = np.round(prediction[pred_label]/np.sum(prediction)*100,2)
            cv2.putText(frame2, f'Confidence: {confidence}%', (1000, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            dcoord = [720, 550]
            cv2.putText(frame2, 'vegan:', (dcoord[0], dcoord[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'vegetarian:', (dcoord[0]-53, dcoord[1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'gluten-free:', (dcoord[0]-69, dcoord[1]+60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'nut-free:', (dcoord[0]-35, dcoord[1]+90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'soy-free:', (dcoord[0]+200, dcoord[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'red meat-free:', (dcoord[0]+200-70, dcoord[1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'shellfish-free:', (dcoord[0]+200-56, dcoord[1]+60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(frame2, 'dairy-free:', (dcoord[0]+200-18, dcoord[1]+90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            
            xshift = 80
            yshift = 19
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[0]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[1]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[2]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[3]]
            xshift = 80 + 235
            yshift = 19
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[4]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[5]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[6]]
            yshift = yshift - 30
            frame2[dcoord[1]-yshift:dcoord[1]+msize-yshift, dcoord[0]+xshift:dcoord[0]+msize+xshift, :] = img_yn[restrictions[7]]
            
        cv2.putText(frame2, 'Press [Space Bar] to identify item', 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
        cv2.imshow('Webcam 1',frame2)
            
    else:
        print("Error: Unable to capture frame.")
        
plt.plot(lassopred0 + FCNNpred0.flatten() + lassopred1 + FCNNpred1.flatten())