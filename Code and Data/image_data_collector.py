import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2

def resize_and_position_window(window_name, width, height, x_position, y_position):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, x_position, y_position)
        
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
c = 0
imgHeight = 140
imgWidth = 105

while True:    
    ret0, frame0 = cap0.read()
    frame1 = frame0
        
    if ret0:
        text = f"{c}"
        cv2.putText(frame1, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow('Webcam 1', frame1)
        
        key = cv2.waitKey(1)
        
        if key == 27: # esc
            cap0.release()
            cv2.destroyAllWindows()
            break
        
        if key == 32: # space bar
            img0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            img0 = cv2.resize(img0[:, 80:560,:], (300,300))
            img0 = np.expand_dims(img0, axis = 0)
            
            if not ret0:
                print("Error reading frame from webcam 1")
                break
            
            if c == 0:
                X = img0
            else:
                X = np.vstack((X, img0))
            c += 1
            
            print(f"Num images: {c}")
            
    else:
        print("Error: Unable to capture frame.")

plt.imshow(X[0])
plt.show()
plt.imshow(X[36])
plt.show()
plt.imshow(X[61])
plt.show()
plt.imshow(X[77])
plt.show()
plt.imshow(X[102])
plt.show()
plt.imshow(X[138])
plt.show()
plt.imshow(X[163])
plt.show()
plt.imshow(X[179])
plt.show()
plt.imshow(X[204])
plt.show()
plt.imshow(X[240])
plt.show()
plt.imshow(X[265])
plt.show()
plt.imshow(X[281])
plt.show()
plt.imshow(X[306])
plt.show()
plt.imshow(X[342])
plt.show()
plt.imshow(X[367])
plt.show()
plt.imshow(X[383])
plt.show()

showall = 0
if showall == 1:
    for i in range(len(X)):
        plt.imshow(X[i])
        plt.title(i+1)
        plt.show()

save_file = 1
if c > 0 and save_file == 1:
    save_path = '/Users/Austin Kwon/Desktop/Price Checker V2/item_images_300x300/'
    item_name = 'Vitamin_Code_Grow_Bone_Test'
    np.savez(save_path + item_name+'.npz', X = X)