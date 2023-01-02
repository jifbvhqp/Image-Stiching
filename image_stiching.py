# -*- coding: utf-8 -*-
import numpy as np
import cv2
from module import showImg,Find_Matches,FeatureMatchingImg,GetPointFromImg,homomat,warp,trim,nearest_neighbor_blending
#讀取圖片
picture_name1 = 'hill1.jpg'
picture_name2 = 'hill2.jpg'

img1_Ori = cv2.imread('./data/'+ picture_name1 , cv2.IMREAD_COLOR)
img2_Ori = cv2.imread('./data/'+ picture_name2 , cv2.IMREAD_COLOR)

img1 = cv2.imread('./data/'+ picture_name1 , cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/'+ picture_name2 , cv2.IMREAD_GRAYSCALE)

#使用OpenCV SIFT分別獲得兩張圖片的特徵點和特徵向量
# sift.detectAndCompute() return 
# keypoint : class
#	- pt : list [x,y] keypoints coordinates.
#	- size : diameter of the meaningful keypoint neighborhood.
#	- angle : computed orientation of the keypoint (-1 if not applicable).
#	- response : the response by which the most strong keypoints have been selected.
#	- octave : octave (pyramid layer) from which the keypoint has been extracted.
#	- class_id : object class
# descriptor : list
#	- Computed descriptors, eigen vectors
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#根據兩張圖片的特徵向量，計算L2距離，去獲取兩張圖片最相近的對應特徵點的索引 : [idx1,idx2] -> idx1 : for kp1 , idx2 : for kp2
matches = Find_Matches(des1,des2,ratio = 0.5,k=2)

#左右合併兩張圖片為一張圖片，根據兩張圖片的對應點繪製連線
Feature_img = FeatureMatchingImg(img1_Ori,img2_Ori,matches,kp1,kp2)

#取得兩張圖片對應的三維座標點
points_in_img1,points_in_img2 = GetPointFromImg(kp1,kp2,matches)

#取得Homography矩陣
H = homomat(points_in_img2, points_in_img1)

#根據Homography矩陣將第二張圖片warp到第一張圖片的座標空間，並且合併兩張圖片
res = warp(img1_Ori,img2_Ori,H)
res = trim(res)

#在根據Homography矩陣合併圖片後，第二張圖片(右邊)部分會有黑色網狀情形出現，此時使用nearest neighbor做blending去處理此情形
final_img = nearest_neighbor_blending(res)
			
#秀出結果
cv2.imshow("Feature_img", Feature_img/255)		  
cv2.imshow("original_image_stitched_crop.jpg", final_img/255)
showImg(final_img)
print("Done")



