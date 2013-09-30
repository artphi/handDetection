import pickle, cv2
import threading
import numpy as np
import sys
from faceDetection import FaceDetection as fd
class HDproject(object):

	## ------------------------------------------------------------
	# Initialization 
	def __init__(self):

		self.debug = False
		if len(sys.argv) == 2:
			if sys.argv[1] == 'debug':
				self.debug = True
				print "Debug mode"
		#Loading from .config
		try:  self.Vars = pickle.load(open(".config", "r"))
		except:
			print "Config file (\".config\") not found."
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
		self.cap = cv2.VideoCapture(0)
		self.thread = np.zeros((60,60),np.uint8)
		self.faceD = np.zeros((60,60),np.uint8)
		self.turn = 0
		#Start
		a = fd(self)

		a.start()
		self.run()
		a.stop()
		cv2.destroyAllWindows()

	## ------------------------------------------------------------
	def run(self):
		run = True
		while(run):
			ret, self.orig_im = self.cap.read()
			self.orig_im = cv2.flip(self.orig_im,1)
			self.thread = self.orig_im.copy()
				#faceDetect = self.faceDetection(self.orig_im)
				
			self.turn += 1
			self.turn = self.turn%10
			skin = self.skinExtraction(self.yCrCbConversion(self.orig_im))
			skin = cv2.bitwise_not(skin)
			#skin = cv2.add(skin,self.faceD)
			contours = cv2.findContours(skin,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
			for cnt in contours:
				if cv2.contourArea(cnt) > 10000:
					
					hull = cv2.convexHull(cnt,returnPoints = False)
					
					defects = cv2.convexityDefects(cnt,hull)
					try:
						for i in range(defects.shape[0]):
							s,e,f,d = defects[i,0]
							start = tuple(cnt[s][0])
							end = tuple(cnt[e][0])
							far = tuple(cnt[f][0])
							cv2.line(self.orig_im,start,end,[0,255,0],2)
							cv2.circle(self.orig_im,far,5,[0,0,255],-1)
					except: pass
					moments = cv2.moments(cnt)
					try:
						centroid_x = int(moments['m10']/moments['m00'])
						centroid_y = int(moments['m01']/moments['m00'])
						cv2.circle(self.orig_im, (centroid_x, centroid_y), 20, (0,255,255), 10)
					except Exception as detail: print detail

			# Debuging tools
			if self.debug:
				cv2.drawContours(self.orig_im,[cnt],-1,(0,255,0),-1)
			# output video
			cv2.imshow('outPut',self.orig_im)

			if cv2.waitKey(1) & 0xFF == ord('q'):
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
		except: pass

	## ------------------------------------------------------------
	def backGroundDetection(self):
		pass

	## ------------------------------------------------------------
	def channelsSplit(self,image):
		return cv2.split(image)
	## ------------------------------------------------------------
	def thresolding(self,image, min_th, max_th):
		val, mask = cv2.threshold(image, min_th, 255, cv2.THRESH_BINARY)
		val, mask_inv = cv2.threshold(image, max_th,255, cv2.THRESH_BINARY_INV)
		return cv2.add(mask, mask_inv)
	## ------------------------------------------------------------
	def binarization(self):
		pass

	## ------------------------------------------------------------
	def morphology(self):
		pass

	## ------------------------------------------------------------
	def faceRemoval(self):
		pass
	## ------------------------------------------------------------
	def cannyEdgeSegmentation(self):
		pass

	## ------------------------------------------------------------
	def skinExtraction(self,image):
		y_img = self.thresolding(image[0],self.th_Y_min, self.th_Y_max)
		cr_img = self.thresolding(image[1],self.th_CR_min, self.th_CR_max)
		cb_img = self.thresolding(image[2],self.th_CB_min, self.th_CB_max)
		final = cv2.add(cb_img,cr_img)
		
		try:
			final = cv2.add(final,self.faceD)
		except Exception as detail: print "error on mask ", detail

		
		
		
		#final = cv2.erode(final, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))   
		#final = cv2.dilate(final,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
		if self.debug:
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

