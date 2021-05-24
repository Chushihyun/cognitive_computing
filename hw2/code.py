import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance

def feature_color_extract(img):
    img=cv2.resize(img,(320,320))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    mask = np.zeros((320,320), np.uint8)
    # only use middle of image
    for i in range(50,270):
        for j in range(50,270):
            # weed out all black or white pixel
            if img[i][j][0]>=2 and img[i][j][0]<=254:
                mask[i][j] = 255

    feature_mean=cv2.mean(img,mask=mask)
    return feature_mean[0:3]

def collect_filter():
    filters=[]
    for scale in [10,25,40]:
        for theta in range(6):
            kernel=cv2.getGaborKernel((50,50),scale,math.pi*float(theta)/6,10,0.5,0)
            filters.append(kernel)
    return filters
   

def feature_shape_extract(img):
    img=cv2.resize(img,(320,320))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gabor filter
    filters=collect_filter()
    # calculate each filter results' mean and std as feature
    feat=[]
    for filt in filters:
        img_filt=cv2.filter2D(img,-1,filt)
        img_filt=np.power(img_filt,2)
        mean=np.mean(img_filt)
        std=np.std(img_filt)
        feat.append(mean)
        feat.append(std)
    return feat
    
def feature_local_extract(img):
    img=cv2.resize(img,(320,320))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use SIFT to get characteristic point
    sift=cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors=sift.detectAndCompute(img,None)
    return keypoints,descriptors

######  process data  #############

cate_table=[]
feature_color=np.zeros([35,20,3])
feature_shape=np.zeros([35,20,36])
feature_local=np.zeros([35,20,2],dtype=np.ndarray)
index0=0
index1=0
path="./HW2-database-20f/database"

# get feature
for cate in os.listdir(path):
    if cate==".DS_Store":
        continue
    cate_table.append(cate)
    path_cate=path+"/"+cate
    for name in os.listdir(path_cate):
        path_name=path_cate+"/"+name
        img=cv2.imread(path_name)
        # color feature
        feature_color[index0][index1]=feature_color_extract(img)
        # shape feature
        feature_shape[index0][index1]=feature_shape_extract(img)
        # local feature
        keypoints,descriptors=feature_local_extract(img)
        feature_local[index0][index1][0]=keypoints
        feature_local[index0][index1][1]=descriptors
        
        index1+=1
    index1=0
    index0+=1
    
    print("finish %s"%cate)

###############################

def evaluate_color_ap(all,i,j):
    dist=np.zeros([35,20])
    for x in range(all.shape[0]):
        for y in range(all.shape[1]):
            if (x,y)!=(i,j):
                d=distance.cosine(all[i][j],all[x][y])
                dist[x][y]=d
    dist_list=[(x,y,dist[x][y]) for x in range(35) for y in range(20)]
    dist_list.sort(key = lambda s: s[2])
    cnt=0.0
    total=0.0
    for index in range(len(dist_list)):
        if dist_list[index][0]==i and index!=0:
            cnt+=1
            total+=float(cnt/float(index))
    return total/19

def evaluate_shape_ap(all,i,j):
    dist=np.zeros([35,20])
    for x in range(all.shape[0]):
        for y in range(all.shape[1]):
            if (x,y)!=(i,j):
                d=distance.euclidean(all[i][j],all[x][y])
                dist[x][y]=d
    dist_list=[(x,y,dist[x][y]) for x in range(35) for y in range(20)]
    dist_list.sort(key = lambda s: s[2])
    cnt=0.0
    total=0.0
    for index in range(len(dist_list)):
        if dist_list[index][0]==i and index!=0:
            cnt+=1
            total+=float(cnt/float(index))
    return total/19
    
    
def matched_percentage(a,b):
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(a[1], b[1], k=2)
    # determine number of "good" match
    good=0
    for m, n in matches:
        if m.distance < 0.35*n.distance:
            good+=1
    percent=float(good)/float(len(a[0]))
    return percent
          
# another match algorithm
"""
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(a[1],b[1],k=2)
    good = 0
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good+=1
    percent=(good)/len(a[0])
    return percent
"""


def evaluate_local_ap(all,i,j):
    similarty=np.zeros([35,20])
    for x in range(all.shape[0]):
        for y in range(all.shape[1]):
            if (x,y)!=(i,j):
                if len(all[x][y][0])==0 or len(all[i][j][0])==0:
                    similarty[x][y]=0
                    continue
                s=matched_percentage(all[i][j],all[x][y])
                similarty[x][y]=1-s
                
            
    dist_list=[(x,y,similarty[x][y]) for x in range(35) for y in range(20)]
    dist_list.sort(key = lambda s: s[2])
    cnt=0.0
    total=0.0
    for index in range(len(dist_list)):
        if dist_list[index][0]==i and index!=0:
            cnt+=1
            total+=float(cnt/float(index))
    
    return total/19
    
def evaluate_fusion_ap(local,color,i,j):
    similarty=np.zeros([35,20])
    for x in range(local.shape[0]):
        for y in range(local.shape[1]):
            if (x,y)!=(i,j):
                d=distance.cosine(color[i][j],color[x][y])
                similarty+=d
                if len(local[x][y][0])==0 or len(local[i][j][0])==0:
                    similarty[x][y]+=0
                    continue
                s=matched_percentage(local[i][j],local[x][y])
                similarty[x][y]+=8*(-s)
                
    dist_list=[(x,y,similarty[x][y]) for x in range(35) for y in range(20)]
    dist_list.sort(key = lambda s: s[2])
    cnt=0.0
    total=0.0
    for index in range(len(dist_list)):
        if dist_list[index][0]==i and index!=0:
            cnt+=1
            total+=float(cnt/float(index))
    return total/19


#######  calculate MAP  ##########


# color
ap_table_color=np.zeros([35,20])
for i in range(35):
    for j in range(20):
        ap_table_color[i][j]=evaluate_color_ap(feature_color,i,j)
map_table_color=np.zeros([36])
for i in range(len(map_table_color)-1):
    map_table_color[i]=np.mean(ap_table_color[i][:])
map_table_color[35]=np.mean(map_table_color[0:35])
print(map_table_color)
np.savez('color_map.npz', a=map_table_color, b=cate_table)


# shape
ap_table_shape=np.zeros([35,20])
for i in range(35):
    for j in range(20):
        ap_table_shape[i][j]=evaluate_shape_ap(feature_shape,i,j)
        
map_table_shape=np.zeros([36])
for i in range(len(map_table_shape)-1):
    map_table_shape[i]=np.mean(ap_table_shape[i][:])
    
map_table_shape[35]=np.mean(map_table_shape[0:35])
print(map_table_shape)
np.savez('shape_map.npz', a=map_table_shape, b=cate_table)


# local
ap_table_local=np.zeros([35,20])
for i in range(35):
    for j in range(20):
        ap_table_local[i][j]=evaluate_local_ap(feature_local,i,j)
    print("finish %s"%cate_table[i])
        
map_table_local=np.zeros([36])
for i in range(len(map_table_local)-1):
    map_table_local[i]=np.mean(ap_table_local[i][:])
map_table_local[35]=np.mean(map_table_local[0:35])
print(map_table_local)

np.savez('local_map.npz', a=map_table_local, b=cate_table)


# fusion

ap_table_fusion=np.zeros([35,20])
for i in range(35):
    for j in range(20):
        ap_table_fusion[i][j]=evaluate_fusion_ap(feature_local,feature_color,i,j)
    print("finish %s"%cate_table[i])
        
map_table_fusion=np.zeros([36])
for i in range(len(map_table_fusion)-1):
    map_table_fusion[i]=np.mean(ap_table_fusion[i][:])
map_table_fusion[35]=np.mean(map_table_fusion[0:35])
print(map_table_fusion)
np.savez('fusion_map.npz', a=map_table_fusion, b=cate_table)

print(cate_table)

####################################
