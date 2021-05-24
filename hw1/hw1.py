import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def adjust(i,j,len,wid):
# check if out of boundary
    if i<0:
        x=0
    elif i>=0 and i<len:
        x=i
    elif i>=len:
        x=len-1
    if j<0:
        y=0
    elif j>=0 and j<wid:
        y=j
    elif j>=wid:
        y=wid-1
    return x,y

def pixel_convolution(img,filter,i,j,k):
    total=0
    count=0
    for x,y in filter:
        m,n=adjust(i+x,j+y,img.shape[0],img.shape[1])#avoid out of boundary
        total+=img[m][n][k]
        count+=1
    
    return total/count

def conv(img):
    filter=[]
    # size:5*5
    for i in range(-2,3,1):
        for j in range(-2,3,1):
            filter.append((i,j))

    output1=np.zeros(img.shape,np.uint8)
    output2=np.zeros(img.shape,np.uint8)
    c=0.6
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                low=pixel_convolution(img,filter,i,j,k)
                output1[i][j][k]=low
                original=img[i][j][k]
                result=c/(2*c-1)*original-(1-c)/(2*c-1)*low
                output2[i][j][k]=max(min(result,255),0) #avoid overflow
        if(i%100==0):
            print("finish %d"%(i))
    return (output1,output2)

def add_text(img):
    font=cv2.FONT_HERSHEY_SIMPLEX
    position=(40,2970)
    fontScale=3
    fontColor=(255,255,255)
    lineType=2

    cv2.putText(img,'R09922115',
        position,
        font,
        fontScale,
        fontColor,
        lineType,
        cv2.LINE_AA)
    return img

###########

img=cv2.imread("input.png")
img1,img2=conv(img)
img1=add_text(img1)
img2=add_text(img2)
cv2.imwrite("smooth.png",img1)
cv2.imwrite("sharpen.png",img2)
