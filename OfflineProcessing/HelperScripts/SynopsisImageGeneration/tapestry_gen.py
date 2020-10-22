import cv2
import numpy as np
import math
from random import randint
from numpy import savetxt
import os
import sys

color_image_dir=sys.argv[1]
binary_image_dir=sys.argv[2]
color_images = [color_image_dir+"/"+f for f in os.listdir(color_image_dir)]
color_images.sort()
binary_images= [binary_image_dir+"/"+f for f in os.listdir(binary_image_dir)]
binary_images.sort()


height = 288
width = 352
m=4
synopsis_image = np.zeros((height*m,width*m,3), np.uint8)
photo_no=np.zeros((height*m,width*m))
video_no=np.zeros((height*m,width*m))
H_C=height*m
W_C=width*m
H=0
W=0
m_W=100000000
count=0
for i in range(len(color_images)):
	temp_string=color_images[i].split('/')
	temp_string=temp_string[len(temp_string)-1]
	temp_string=temp_string.split('.')[0]
	video_id=temp_string.split('_')[1]
	frame_id=temp_string.split('_')[2]
	color=cv2.imread(color_images[i])
	binary=cv2.imread(binary_images[i],0)
	ret,binary = cv2.threshold(binary,100,255,cv2.THRESH_BINARY)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(binary,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations=1)
	image, contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	m_a=0
	x,y,w,h=0,0,0,0
	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		if area>m_a:
			m_a=area
			x,y,w,h = cv2.boundingRect(contours[i])
	y=0 #For now
	w=max(w,36)
	if x+w>width:
		w=width-x
	count=count+1
	if count==len(color_images):
		if W+w<m_W and W+w<=W_C:
			m_W=W+w
	if W+w>W_C:
		if W<m_W:
			m_W=W
		W=0
		H=H+height
	if H==H_C:
		break
	synopsis_image[H:H+height,W:W+w,:]=color[0:height,x:x+w]
	photo_no[H:H+height,W:W+w]=frame_id
	video_no[H:H+height,W:W+w]=video_id
	cv2.rectangle(color,(x,0),(x+max(w,72),y+height), (255,0,0), 2)
	W=W+w

	cv2.imshow("Color",color)
	cv2.imshow("Binary",binary)
	cv2.imshow("Synopsis",synopsis_image)
	cv2.waitKey(0)


synopsis_image=synopsis_image[0:H+height,0:m_W]
video_no=video_no[0:H+height,0:m_W]
photo_no=photo_no[0:H+height,0:m_W]



h,w,cols=synopsis_image.shape

compressed_height=384

photo_meta=np.zeros((compressed_height,640))
video_meta=np.zeros((compressed_height,640))

compressed_image=np.zeros((compressed_height,640,3),np.uint8)
for i in range(compressed_height):
	for j in range(640):
		compressed_image[i][j]=synopsis_image[int(h*i/compressed_height)][int(w*j/640)]
		photo_meta[i][j]=int(photo_no[int(h*i/compressed_height)][int(w*j/640)])
		video_meta[i][j]=int(video_no[int(h*i/compressed_height)][int(w*j/640)])

cv2.imshow("Synopsis",compressed_image)

cv2.imshow("Synopsis Image",compressed_image)

cv2.imwrite("./synopsis_image.jpg",compressed_image)


cv2.waitKey(0)

		# hull = cv2.convexHull(contours[i])

print "var frame=",

print "[",
for x in photo_meta:
	print "[",
	print x[0],
	for i in range(1,len(x)):
		print ",",x[i],
	print "],",
print "]"
	
print "var video_no=",

print "[",
for x in video_meta:
	print "[",
	print x[0],
	for i in range(1,len(x)):
		print ",",x[i],
	print "],",
print "]"