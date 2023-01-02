import numpy as np
import cv2
def showImg(img):
	sigma = 75
	cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
	#blur = cv2.bilateralFilter(img,9,sigma,sigma)

	cv2.imshow('My Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def Find_Matches(des1,des2,ratio = 0.1,k=2):
	res = []
	for v1 in des1:
		list_distance = []
		for v2 in des2:
			dis = np.sqrt((np.square(v1-v2)).sum())
			list_distance.append(dis)
		list_distance = np.array(list_distance)
		Sort_index = np.argsort(list_distance) 
		match = []
		for K in range(k):
			Index = Sort_index[K]
			match.append([list_distance[Index],Index])	#distance,index
		res.append(match)
	#Pick good
	good = []
	for (i,(m,n)) in enumerate(res):
		if m[0] < ratio * n[0]:		  #distance
			good.append([i,m[1]])	#queryIdx,trainIdx
	return good
				
def FeatureMatchingImg(img1,img2,matches,kp1,kp2):
	connect_img = np.hstack((img1,img2))
	for (queryIdx,trainIdx) in matches:
		p1 = np.array(kp1[queryIdx].pt)
		p2 = np.array(kp2[trainIdx].pt)
		p2[0] += img1.shape[1]

		Img1RowIdx = p1[1]
		Img1ColIdx = p1[0]
		Img1Index = np.array([Img1RowIdx,Img1ColIdx])
		Img2RowIdx = p2[1]
		Img2ColIdx = p2[0]
		Img2Index = np.array([Img2RowIdx,Img2ColIdx])
		direction = np.array([Img2RowIdx-Img1RowIdx,Img2ColIdx-Img1ColIdx])
		direction = direction/np.linalg.norm(direction)
		k = 0
		color = np.random.rand(3)*255
		while True:
			curren_piexl = Img1Index + (k*direction).astype(int)
			k+=1
			if curren_piexl[1]> Img2Index[1]:
				break
			connect_img[int(curren_piexl[0]),int(curren_piexl[1]),:] = color
	return connect_img

def getHi(p1,p2):
	A = []
	for i in range(0, len(p1)):
		x, y = p1[i][0], p1[i][1]
		u, v = p2[i][0], p2[i][1]
		A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
		A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
	A = np.asarray(A)
	U, S, Vh = np.linalg.svd(A)
	L = Vh[-1,:] / Vh[-1,-1]
	Hi = L.reshape(3, 3)
	return Hi

def ComputeInliers(Hi,t,img1_p,img2_p,unchoice_idx):
	res = 0
	for index in unchoice_idx:
		p0 = img1_p[index]
		p1 = img2_p[index]
		p2 = Hi.dot(p0.T)
		Score = np.linalg.norm(p2.T - p1)
		if(Score<=t):
			res+=1
	return res

def GetPointFromImg(kp1,kp2,matches):
	dst__pts = []
	src__pts = []
	for matche in matches:
		p1 = np.float32( kp1[matche[0]].pt + (1,) )
		p2 = np.float32( kp2[matche[1]].pt + (1,) )
		dst__pts.append(p1)
		src__pts.append(p2)
	points_in_img1 = np.array(dst__pts)
	points_in_img2 = np.array(src__pts)
	return points_in_img1,points_in_img2

def homomat(src__pts,dst__pts):
	sigma = 2
	t = np.sqrt(5.99*sigma*sigma)
	s = 8	# Minimum number needed to fit the model
	p = 0.95 #prob=0.95
	e = 0.05 #outlier ratio: e
	N = np.log(1-p)/np.log(1-(1-e)**s)
	k = int(N) # number of iterations
	H = np.eye(3)
	MaxInliers = -1
	Index = np.arange(len(src__pts))	#[0,1,2,...]
	for i in range(k):
		np.random.shuffle(Index)
		choice_idx = Index[0:s]
		unchoice_idx = Index[s:]
		p1 = np.array([ src__pts[index] for index in choice_idx])
		p2 = np.array([ dst__pts[index] for index in choice_idx])
		Hi = getHi(p1,p2)
		nInliers = ComputeInliers(Hi,t,src__pts,dst__pts,unchoice_idx)
		if MaxInliers < nInliers:
			MaxInliers = nInliers
			H = Hi
	#print(MaxInliers)
	return H
	
def warp(img1_Ori,img2_Ori,H):
	res = np.zeros((img1_Ori.shape[0] , img1_Ori.shape[1] + img2_Ori.shape[1],3))
	res[:] = np.nan
	for row in range(img2_Ori.shape[0]):
		for col in range(img2_Ori.shape[1]):
			#p = np.array([row,col,1])
			p = np.array([col,row,1])
			new_p = (H.dot(p.T)).T
			new_p = new_p/new_p[-1]
			new_p = new_p.astype(int)
			
			new_p[0] = min(max(0,new_p[0]),res.shape[1]-1)	 
			new_p[1] = min(max(0,new_p[1]),res.shape[0]-1)
			res[new_p[1],new_p[0],:] = img2_Ori[row,col,:]

	for i in range(img1_Ori.shape[0]):
		for j in range(img1_Ori.shape[1]):
			res[i][j] = img1_Ori[i][j]

	return res

def trim(frame):
	cols = np.isnan(frame).all(axis=0)
	rows = np.isnan(frame).all(axis=1)
	cols = cols[: , 0]
	rows = rows[: , 0]
	x = cols.size - 1
	y = rows.size - 1
	for i in reversed(range(cols.size)):
		if(not cols[i]):
			x = i
			break
	for i in reversed(range(rows.size)):
		if(not rows[i]):
			y = i
			break	 
	return frame[:y , :x]

def blending(r,c,res):
	x1 = x2 = y1 = y2 = np.nan
	for i in range(c , -1 , -1):
		if( np.isnan(res[r , i , 0])):
			continue
		else:
			x1 = i
			break
	for i in range(c , res.shape[1] , 1):
		if( np.isnan(res[r , i , 0])):
			continue
		else:
			x2 = i
			break
		
	for i in range(r , -1 , -1):
		if( np.isnan(res[i , c , 0])):
			continue
		else:
			y1 = i
			break
	for i in range(r , res.shape[0] , 1):
		if(np.isnan(res[i , c , 0])):
			continue
		else:
			y2 = i
			break
	
	if(np.isnan(x1) or np.isnan(x2)):
		if(np.isnan(y1) or np.isnan(y2)):
			return np.nan
		else:
			Ry = ((y2 - r) / (y2-y1)) * res[y1 , c] + ((r - y1) / (y2-y1)) * res[y2 , c]
			return (Ry).astype(int)
	elif(np.isnan(y1) or np.isnan(y2)):
		Rx = ((x2 - c) / (x2-x1)) * res[r , x1] + ((c - x1) / (x2-x1)) * res[r , x2]
		return (Rx).astype(int)
	else:		
		Rx = ((x2 - c) / (x2-x1)) * res[r , x1] + ((c - x1) / (x2-x1)) * res[r , x2]
		Ry = ((y2 - r) / (y2-y1)) * res[y1 , c] + ((r - y1) / (y2-y1)) * res[y2 , c]
		return ((Rx + Ry) / 2).astype(int)

def nearest_neighbor_blending(img):	
	final_img = np.zeros(img.shape)
	for r in range(img.shape[0]):
		for c in range(img.shape[1]):
			if np.isnan(img[r , c , 0]):
				final_img[r,c,:] = blending(r,c,img)  #just neatest neighbor
				img[r,c,:] = final_img[r,c,:]
			else:
				final_img[r,c,:] = img[r,c,:]
	final_img = final_img.astype(np.uint8)	
	return final_img