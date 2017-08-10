"""
Essentially, this program processes the dataset to write to a biometric file.
A Separate program takes an image input and compares the biometrics of that face to the biometric file
and returns a set of images that passes a certain threshold.
"""
#based on this tutorial: https://www.youtube.com/watch?v=88HdqNDQsEk
#CUDA Programming Python: https://developer.nvidia.com/how-to-cuda-python
import cv2
import sys
import os
import numpy as py
import time
import threading

#grab command line arguments
#file_name = sys.argv[1]
#cascade files http://alereimondo.no-ip.org/OpenCV/34
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier('haar/ojoI.xml')
reye_cascade = cv2.CascadeClassifier('haar/ojoD.xml')
nose_cascade = cv2.CascadeClassifier('haar/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('haar/Mouth.xml')

num_threads = int(sys.argv[1])
mutex = threading.Lock()
bio_file = open('biometrics.txt', 'w')
directory = "2_All"

"""
REQUIREMENTS
Images must be normalized to 100x100px and grayscaled to consistency.
All image must be front-faced or else more inaccuracies.

The naive-biometrics will be calculated and stored as follows:
Left eye distance from sides of face / size of face (length and width)
[file_name, ["leye",[leftDist, rightDist, topDist, downDist, eyeDist]],["reye", [...], ...]
"""
def measureBiometrics(face, reye, leye, mouth, nose, file_name, num):
	measurement = []
	measurement.append(file_name)
	x, y ,w, h = face[0] #only regarding one face

	#for simplicity sake, only measure the first instance of each feature.
	#inaccuracies will happen.
	for (rx,ry,rw,rh) in reye:
		measurement.append(rw)
		measurement.append(rh)
		break
	
	for (lx,ly,lw,lh) in leye:
		measurement.append(lw)
		measurement.append(lh)
		break

	for (mx,my,mw,mh) in mouth:
		measurement.append(mw)
		measurement.append(mh)
		break
	for (nx,ny,nw,nh) in nose:
		measurement.append(nw)
		measurement.append(nh)
		break

	mutex.acquire()
	for i in measurement:
		bio_file.write(str(i) + " ")

	bio_file.write('\n')	
	mutex.release()
	return

"""
For optimization purposes, the program should preprocess the dataset to one image buffer.
This will greatly reduce the runtime of the program as each thread will skip the read/write step.
"""
def preprocess(image):
	return

def thread_func(rank, num_files):
	offset = int(((num_files/num_threads)*rank))
	end = int(offset+(num_files/num_threads))
	num = 0
	for i in range(offset, end):
		file_name = "{0}/{1}.jpg".format(directory, i)
		#print("Thread {0} processing {1}...".format(rank, file_name))
		gray = cv2.imread(file_name, 0)
		#img = cv2.resize(img, (100, 100)) #to keep the image consistent.
		#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		for (x,y,w,h) in faces:
			#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			#roi_color = img[y:y+h, x:x+w]
			#clean up image buffers to optimize space requirement.

			reye = reye_cascade.detectMultiScale(roi_gray)
			leye = leye_cascade.detectMultiScale(roi_gray)
			nose = nose_cascade.detectMultiScale(roi_gray)
			mouth = mouth_cascade.detectMultiScale(roi_gray)

			valid = (len(reye)*len(leye)*len(nose)*len(mouth))

			"""
			for (rx,ry,rw,rh) in reye:
				cv2.rectangle(roi_color,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)

			for (lx,ly,lw,lh) in leye:
				cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)

			for (mx,my,mw,mh) in mouth:
				cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)

			for (nx,ny,nw,nh) in nose:
				cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(120,120,0),2)
			"""

			if (valid > 0): #needs at least all of the features detected in order to properly measure.		
				measureBiometrics(faces, reye, leye, mouth, nose, file_name, num)
				num = num + 1

		#cv2.imwrite(('output_images/{0}.pgm').format(i),img)
	
	print("Thread {0} is done!".format(rank))
		
def main():
	num = 1
	start_time = time.time()
	num_files = len(os.listdir(directory))
	threads = []

	print("Initiating {0} threads...".format(num_threads))
	for i in range(0, num_threads):
		threads.append(threading.Thread(target = thread_func, args=(i,num_files)))
	print("Done!")

	
	for i in range(0, num_threads):
		print("Running thread {0}...".format(i))
		threads[i].start()
	
	for i in range(0, num_threads):
		threads[i].join()

	print("Runtime: %s seconds" %(time.time()-start_time))

	bio_file.close()

if __name__ == "__main__":
	main()
