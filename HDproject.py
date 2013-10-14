#-*- coding: utf-8 -*-

"""-----------------------------------------------------------------------------
HDproject
Hand detection with OpenCV

Autors: 	Aude Piguet
			Olivier Francillon
			Raphael Santos
Due date:	18.11.2013
Release:

Notes:
Please refer to the french document "HDproject - Rapport"

Prerequists:
	* Python 2.7
	* OpenCV 2.4.6.1

Usage
On linux:
	* $ python HDproject.py [config [path]][debug [face, mask, all]]
	* To stop the program, please use the 'q' key, then on the terminal choose if you 
	  want to save or not the modifications
	* 

Files description
	* HDproject.py: Main Class
	* faceDetection.py: Face detection class using haar. Threaded in main class
	* .config: Config file
	* haar: folder containing some haar XML

/--------------------------------------------------------------/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
/--------------------------------------------------------------/


-----------------------------------------------------------------------------"""

import pickle, cv2
import numpy as np
from numpy import sqrt, arccos, rad2deg
import sys
from faceDetection import FaceDetection as fd
class HDproject(object):

	## ------------------------------------------------------------
	# Initialization 
	def __init__(self):

		self.debug = False
		self.debugFace = False
		self.debugMask = False
		self.conf = ".config"
		if len(sys.argv) > 1:
			confArg = False
			for deb in sys.argv :
				if confArg:
					self.conf = deb
					confArg = False
				if deb == 'config':
					confArg = True
				if deb == 'debug':
					self.debug = True
				if deb == 'face' and self.debug == True:
					self.debugFace = True
				if deb == 'mask' and self.debug == True :
					self.debugMask = True
				if deb == 'all' and self.debug == True:
					self.debugFace = True
					self.debugMask = True
				


				print "Debug mode"
		#Loading from .config
		try:  self.Vars = pickle.load(open(self.conf, "r"))
		except:
			print "Config file (" + self.conf + ") not found."
			exit()

		self.th_Y_min = self.Vars["th_Y_min"]
		self.th_Y_max = self.Vars["th_Y_max"]
		self.th_CR_min = self.Vars["th_CR_min"]
		self.th_CR_max = self.Vars["th_CR_max"]
		self.th_CB_min = self.Vars["th_CB_min"]
		self.th_CB_max = self.Vars["th_CB_max"]
		self.blur = self.Vars["blur"]

		#TrackBars
		cv2.namedWindow("Filters")
		cv2.createTrackbar('Y_min','Filters',self.th_Y_min,234,self.onChange_th_Y_min)
		cv2.createTrackbar('Y_max','Filters',self.th_Y_max,234,self.onChange_th_Y_max)
		cv2.createTrackbar('CR_min','Filters',self.th_CR_min,240,self.onChange_th_CR_min)
		cv2.createTrackbar('CR_max','Filters',self.th_CR_max,240,self.onChange_th_CR_max)
		cv2.createTrackbar('CB_min','Filters',self.th_CB_min,240,self.onChange_th_CB_min)
		cv2.createTrackbar('CB_max','Filters',self.th_CB_max,240,self.onChange_th_CB_max)
		cv2.createTrackbar('Blur','Filters',self.blur,100,self.onChange_blur)
		

		#Camera Capture
		if self.debug:
			print "DEBUG: Acquiring video"
		try:
			self.cap = cv2.VideoCapture(0)
			if not self.cap.isOpened():
				print "Cam not found. exiting..."
				sys.exit(1)
		except Exception as detail:
			print "Error Initialization of the camera: ", detail
			sys.exit(1)
		cv2.waitKey(500)
		self.thread = np.zeros((60,60),np.uint8)
		self.faceD = np.zeros((60,60),np.uint8)

		#Start
		self.a = fd(self)

		#a.start()
		self.run()
		self.a.stop()

	## ------------------------------------------------------------
	# main function
	def run(self):

		ret = None
		iteration = 0
		fingers = 0
		while not ret:
			ret,self.orig_im = self.cap.read()
			iteration += 1
			if iteration > 600:
				print "ERROR reading video capture: timeout"
				sys.exit(1)

		self.a.start()
		run = True

		while(run):
			# get the video frame from main camera
			try:
				ret, self.orig_im = self.cap.read()
			except Exception as detail:
				print "ERROR reading video capture: ", detail
				sys.exit(1)
			if ret:
				# flip the image horizontally
				self.orig_im = cv2.flip(self.orig_im,1)
				size = self.orig_im.shape
				if size[1] > 640:
					new_size = np.copy(size)
					new_size[1] = 640
					new_size[0] = float(size[0])/(float(size[1])/float(new_size[1]))
					self.orig_im = cv2.resize(self.orig_im, (new_size[1], new_size[0]))

				#Thread for face detection
				try:
					self.thread = self.orig_im.copy()
				except Exception as detail: print detail, "thread"

				skin = self.skinExtraction(self.yCrCbConversion(self.orig_im))
				skin = cv2.bitwise_not(skin)

				# Find the contours
				contours = cv2.findContours(skin,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]


				for cnt in contours:
					# Select the big area
					if cv2.contourArea(cnt) > 10000:
						# For each contour, get the hull
						hull = cv2.convexHull(cnt,returnPoints = False)
						angles = []
						# For each contour, get the defects
						defects = cv2.convexityDefects(cnt,hull)
						if defects == None: return
						try:
							for i in range(defects.shape[0]):
								s,e,f,d = defects[i,0]
								if d > 5000: #If the distance from the defect to the hull is greater than 5000
									start = tuple(cnt[s][0])
									end = tuple(cnt[e][0])
									far = tuple(cnt[f][0])
									angleDefect = self.angle(far, start, end)		#get the angle of the two line outgoing from the defect
									angles.append(angleDefect)		
									if angleDefect < 90:
										cv2.circle(self.orig_im,far,5,[0,0,255],-1)  	#create a dot
										cv2.line(self.orig_im,start,far,[0,255,0],2) 	#Create a line from the defect to the next hull
										cv2.line(self.orig_im,far,end,[0,255,0],2)		#Create a line from the hull to the next defect
										

						except Exception as detail: 
							print "contour error: ",detail
							#pass
							
						
						self.handCenter(cv2.moments(cnt))
						
						b = filter(lambda a:a<90, angles)
						fingers = len(b) + 1
						print "finger = ", fingers
						
					# Debuging tools
					if self.debug:
						cv2.drawContours(self.orig_im,[cnt],-1,(0,255,0),-1)
				# output video
				cv2.putText(self.orig_im, "fingers = " + `fingers`, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255])
				cv2.imshow('outPut',self.orig_im)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					save = 0
					while save != 'y' and save != 'n':
						print ("Do you want to save? (y/n)")
						save = raw_input("> ")
						if save == 'y':
							self.save()
							run = False
						elif save == 'n':
							run = False
			
		# Release the camera
		self.cap.release()

	## ------------------------------------------------------------
	def angle(self, cent, rect1, rect2):
		v1 = (rect1[0] - cent[0], rect1[1] - cent[1])
		v2 = (rect2[0] - cent[0], rect2[1] - cent[1])
		dist = lambda a:sqrt(a[0] ** 2 + a[1] ** 2)
		angle = arccos((sum(map(lambda a, b:a*b, v1, v2))) / (dist(v1) * dist(v2)))
		angle = abs(rad2deg(angle))
		return angle
	## ------------------------------------------------------------
	# Search and show the hand's center of mass
	def handCenter(self, moments):
		try:
			centroid_x = int(moments['m10']/moments['m00'])
			centroid_y = int(moments['m01']/moments['m00'])
			cv2.circle(self.orig_im, (centroid_x, centroid_y), 15, (0,0,255), 1)
			cv2.circle(self.orig_im, (centroid_x, centroid_y), 10, (0,0,255), 1)
			cv2.circle(self.orig_im, (centroid_x, centroid_y), 5, (0,0,255), -1)
		except Exception as detail: print "Hand center error: ",detail

	## ------------------------------------------------------------
	# Save the parameters
	def save(self):
		self.Vars["th_Y_min"] = self.th_Y_min
		self.Vars["th_Y_max"] = self.th_Y_max
		self.Vars["th_CR_min"] = self.th_CR_min
		self.Vars["th_CR_max"] = self.th_CR_max
		self.Vars["th_CB_min"] = self.th_CB_min
		self.Vars["th_CB_max"] = self.th_CB_max
		self.Vars["blur"] = self.blur
		pickle.dump(self.Vars, open(".config", "w"))
		

	## ------------------------------------------------------------
	# Convert the video in YCrCb and split it in three channels
	# Return an array containing the 3 channels 
	def yCrCbConversion(self,image):
		#Try/Catch
		try:
			yCrCb_im = np.copy(image)
			yCrCb_im = cv2.cvtColor(yCrCb_im, cv2.COLOR_BGR2YCR_CB)
			if self.blur == 0:
				self.blur = 1
			yCrCb_im = cv2.blur(yCrCb_im,(self.blur,self.blur))
			splitted = self.channelsSplit(yCrCb_im)
		
			if self.debug:
				pass
				# cv2.imshow('YCrCb',yCrCb_im)
				# cv2.imshow('Y',splitted[0])
				# cv2.imshow('Cr',splitted[1])
				# cv2.imshow('Cb',splitted[2])
			return splitted
		except Exception as detail: 
			print "Error in yCrCbConversion: ",detail
			pass

	## ------------------------------------------------------------
	# not implemented... Maybe one day...
	def backGroundDetection(self):
		pass

	## ------------------------------------------------------------
	# split an image
	def channelsSplit(self,image):
		return cv2.split(image)

	## ------------------------------------------------------------
	# Generate the thresolding for min and max values
	def thresolding(self,image, min_th, max_th):
		val, mask = cv2.threshold(image, min_th, 255, cv2.THRESH_BINARY)
		val, mask_inv = cv2.threshold(image, max_th,255, cv2.THRESH_BINARY_INV)
		return cv2.add(mask, mask_inv)

	## ------------------------------------------------------------
	def morphology(self,image):
		morpho = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))   
		morpho = cv2.dilate(morpho,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
		return morpho

	## ------------------------------------------------------------
	def cannyEdgeSegmentation(self):
		pass

	## ------------------------------------------------------------
	def skinExtraction(self,image):
		try:
			y_img = self.thresolding(image[0],self.th_Y_min, self.th_Y_max)
			cr_img = self.thresolding(image[1],self.th_CR_min, self.th_CR_max)
			cb_img = self.thresolding(image[2],self.th_CB_min, self.th_CB_max)
			final = cv2.add(cb_img,cr_img)
			final = cv2.add(final,self.faceD)
		except Exception as detail: print "error on mask ", detail

		# maybe not useful...
		#final = self.morphology(final)
		
		

		if self.debug and self.debugMask:
			cv2.imshow('y_img_mask',y_img)
			cv2.imshow('cr_img_mask',cr_img)
			cv2.imshow('cb_img_mask',cb_img)
			cv2.imshow('final',final)
		return final
		
	## ------------------------------------------------------------
	# onChange for TrackBars
	def onChange_th_Y_min(self, value):
		self.th_Y_min = value

	def onChange_th_Y_max(self, value):
		self.th_Y_max = value

	def onChange_th_CR_min(self, value):
		self.th_CR_min = value

	def onChange_th_CR_max(self, value):
		self.th_CR_max = value

	def onChange_th_CB_min(self, value):
		self.th_CB_min = value

	def onChange_th_CB_max(self, value):
		self.th_CB_max = value

	def onChange_blur(self, value):
		self.blur = value
		

	## ------------------------------------------------------------
	def gaussianFilter():
		pass

	## ------------------------------------------------------------
	def handDetectionModule():
		pass

	## ------------------------------------------------------------
	## ------------------------------------------------------------
	## ------------------------------------------------------------






run = HDproject()

