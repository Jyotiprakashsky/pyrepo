import cv2
import numpy as np
import time
import pickle
import pandas as pd
from numpy.linalg import norm

classifier = pickle.load(open('KernelSVMClassifier.sav', 'rb'))

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        #print(hist)

        samples.append(hist)
    return np.float32(samples)


def IClassifier(crop_image):
    imageData = cv2.resize(crop_image, (64, 64))
    imar = []
    imar.append(imageData)
    samples = preprocess_hog(imar)
    df = pd.DataFrame(samples)
    result = classifier.predict(df.iloc[:, :].values)
    # training_set.class_indices
    #print(result)
    if result == 1:

        prediction = '1'
        # print(iname + "==" + prediction)
        return 1
    else:
        # cv.imshow(iname,cv2.imread(iname))
        prediction = '0'
        # print(iname + "==" + prediction)
        return 0

bgSubThreshold = 50
learningRate = 0
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res





cap = cv2.VideoCapture(0)
ret, current_frame = cap.read()
current_frame1 = cv2.flip(current_frame, 1)
current_frame = removeBG(current_frame1)

previous_frame = current_frame

while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

    frame_diff = cv2.absdiff(current_frame_gray,previous_frame_gray)
    frame_diff = cv2.blur(frame_diff, (20, 20))
    frame_v = removeBG(frame_diff)
    ret,thresh = cv2.threshold(frame_v,7,205,0)
    im2, contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
     #cv2.drawContours(current_frame_gray, [cnt], 0, (128,255,0), 2)
     x,y,w,h = cv2.boundingRect(cnt)
     cv2.rectangle(current_frame_gray,(x,y),(x+w,y+h),(0,255,255),2)
     crop_img = current_frame_gray[y:y+h, x:x+w]
     val = IClassifier(crop_img)
     if val == 0:
       cv2.imwrite('./dataIMG/'+ str(time.time())+'.png',crop_img)
       cv2.imshow('./dataIMG/'+ str(time.time())+'.png',current_frame)
	 
	 
	
    cv2.imshow('frame diff ',thresh)
    cv2.imshow('frame',current_frame_gray)	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

cap.release()
cv2.destroyAllWindows()