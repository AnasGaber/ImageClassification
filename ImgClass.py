
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import math

#this function for the color quantization
def color_quantization(org_image, c):
	(h, w) = org_image.shape[:2]
	org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)
	org_image = org_image.reshape((org_image.shape[0] * org_image.shape[1], 3))
	clus = MiniBatchKMeans(c)
	labels = clus.fit_predict(org_image)
	img_quant = clus.cluster_centers_.astype("uint8")[labels]
	img_quant = img_quant.reshape((h, w, 3))
	return cv2.cvtColor(img_quant, cv2.COLOR_LAB2BGR)

#this function is to return back the intensity level as a feature vector
def gray_intensity(org_image, size=(32, 32)):
	return cv2.resize(org_image, size).flatten()

#this function is to return back the color histogram
def color_histogram(org_image, bins=(8, 8, 8)):
	hsv_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
	color_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	cv2.normalize(color_hist, color_hist)
	return color_hist.flatten()
#values k for KKN classifier and Euclidean distance measure
#values c for color quantization value
#values
k=1
c=265
l=2

#reading the TrainingSet
print("Reading the TrainingSet...")
TSPaths = list(paths.list_images("C:\\Users\\Anas\\Desktop\\project1\\TrainingSet"))
#storing the returns of the gray_intensity function
grayscaleImg=[]
#storing the returns of color_histogram fuction
colorHist=[]
#storing the labesls of the images
categories=[]

#loading images
for (i, TSPath) in enumerate(TSPaths):
	image = cv2.imread(TSPath)
	fullPath=TSPath.split('\\')
	category = fullPath[-2]


	#split image into blocks for levels
	(h,w)=image.shape[:2]
	fh=math.ceil(h/l)
	fw=math.ceil(w/l)
	for r in range(0,image.shape[0],fh):
		for c in range(0,image.shape[1],fw):
			category = category
			#color quantizing
			#quantimg = color_quantization(image, c)
			#convert image to grayscale
			grayimg = cv2.cvtColor(image[r:r+fh, c:c+fw,:], cv2.COLOR_BGR2GRAY)
			#obtaining the gray intensity vector
			gray_int = gray_intensity(grayimg)
			#obtaining the color histogram vector
			col_hist = color_histogram(image[r:r+fh, c:c+fw,:])
			#storing gray_int result one by one
			grayscaleImg.append(gray_int)
			#storing col_hist result one by one
			colorHist.append(col_hist)
			#storing category one by one
			categories.append(category)

print("----------Done-----------")
#reading the ValidationSet

print("Reading the ValidationSet..")
VSPaths = list(paths.list_images("C:\\Users\\Anas\\Desktop\\project1\\ValidationSet"))
#storing the returns of the gray_intensity function
vgrayscaleImg=[]
#storing the returns of color_histogram fuction
vcolorHist=[]
#storing the labesls of the images
vcategories=[]

#loading images
for (i, VSPath) in enumerate(VSPaths):
	vimage = cv2.imread(VSPath)
	vfullPath=VSPath.split('\\')
	vcategory = vfullPath[-2]
	#split image into blocks for levels
	(h,w)=vimage.shape[:2]
	fh=math.ceil(h/l)
	fw=math.ceil(w/l)
	for r in range(0,vimage.shape[0],fh):
		for c in range(0,vimage.shape[1],fw):
			vcategory = vcategory
			#color quantizing
			#quantimg = color_quantization(image, c)
			#convert image to grayscale
			vgrayimg = cv2.cvtColor(vimage[r:r+fh, c:c+fw,:], cv2.COLOR_BGR2GRAY)
			#obtaining the gray intensity vector
			vgray_int = gray_intensity(vgrayimg)
			#obtaining the color histogram vector
			vcol_hist = color_histogram(vimage[r:r+fh, c:c+fw,:])
			#storing gray_int result one by one
			vgrayscaleImg.append(vgray_int)
			#storing col_hist result one by one
			vcolorHist.append(vcol_hist)
			#storing category one by one
			vcategories.append(vcategory)

print("----------Done-----------")

#Reading TestSet

print("Reading the TestSet...")
SSPaths = list(paths.list_images("C:\\Users\\Anas\\Desktop\\project1\\TestSet"))
#storing the returns of the gray_intensity function
tgrayscaleImg=[]
#storing the returns of color_histogram fuction
tcolorHist=[]
#storing the labesls of the images
tcategories=[]

#loading images
for (i, SSPath) in enumerate(SSPaths):
	timage = cv2.imread(SSPath)
	tfullPath=SSPath.split('\\')
	tcategory = tfullPath[-2]
	#split image into blocks for levels
	(h,w)=image.shape[:2]
	fh=math.ceil(h/l)
	fw=math.ceil(w/l)
	for r in range(0,timage.shape[0],fh):
		for c in range(0,timage.shape[1],fw):
			#obtaining the category name from file path
			tcategory = tcategory
			#color quantizing
			#quantimg = color_quantization(image, c)
			#convert image to grayscale
			tgrayimg = cv2.cvtColor(timage[r:r+fh, c:c+fw,:], cv2.COLOR_BGR2GRAY)
			#obtaining the gray intensity vector
			tgray_int = gray_intensity(tgrayimg)
			#obtaining the color histogram vector
			tcol_hist = color_histogram(timage[r:r+fh, c:c+fw,:])
			#storing gray_int result one by one
			tgrayscaleImg.append(tgray_int)
			#storing col_hist result one by one
			tcolorHist.append(tcol_hist)
			#storing category one by one
			tcategories.append(tcategory)

print("----------Done-----------")
print("Calculating the results....")
#calculation a k-NN classifer on the grayscale intensities
test = KNeighborsClassifier(k)
test.fit(grayscaleImg, categories)
per = test.score(tgrayscaleImg, tcategories)
print("The grayscale intensity accuracy is  {:.2f}%".format(per * 100))

#calculation a k-NN classifer on the color histogram
test = KNeighborsClassifier(k)
test.fit(colorHist, categories)
per = test.score(tcolorHist, tcategories)
print("The color histogram accuracy is {:.2f}%".format(per * 100))
