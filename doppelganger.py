"""
Actual program the user will be using.
Measures the image input and compares values to biometrics.txt

"""
import cv2
import sys
import os
import numpy as py
import operator

#cascade files http://alereimondo.no-ip.org/OpenCV/34
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier('haar/ojoI.xml')
reye_cascade = cv2.CascadeClassifier('haar/ojoD.xml')
nose_cascade = cv2.CascadeClassifier('haar/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('haar/Mouth.xml')

myimage = cv2.imread(sys.argv[1])
biofile = open("biometrics.txt", 'r')
results = open("results.txt", 'w')
threshold = float(sys.argv[2])

def measureBiometrics(face, reye, leye, mouth, nose):
	measurement = []
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

	return measurement

def detectFace():
	#print("Thread {0} processing {1}...".format(rank, file_name))
	gray = myimage
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	measurement = []
	for (x,y,w,h) in faces:
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]

		reye = reye_cascade.detectMultiScale(roi_gray)
		leye = leye_cascade.detectMultiScale(roi_gray)
		nose = nose_cascade.detectMultiScale(roi_gray)
		mouth = mouth_cascade.detectMultiScale(roi_gray)

		for (rx,ry,rw,rh) in reye:
			cv2.rectangle(roi_gray,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)

		for (lx,ly,lw,lh) in leye:
			cv2.rectangle(roi_gray,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)

		for (mx,my,mw,mh) in mouth:
			cv2.rectangle(roi_gray,(mx,my),(mx+mw,my+mh),(0,0,255),2)

		for (nx,ny,nw,nh) in nose:
			cv2.rectangle(roi_gray,(nx,ny),(nx+nw,ny+nh),(120,120,0),2)
				
		measurement = measureBiometrics(faces, reye, leye, mouth, nose)

		cv2.imwrite('analyzed.jpg', gray)
	return measurement

def readFile():
	biometrics = []	

	for line in biofile:
		bioline = []
		line = line.split()
		for word in line:
			bioline.append(word)
		biometrics.append(bioline)
		
	return biometrics

def compare(myStats, other):
	result = []
	my_sum = 0
	for item in myStats:
		my_sum = my_sum + item

	my_sum = my_sum/8
	print(myStats)
	print(my_sum)
	for stat in other:
		ref_sum = 0
		
		for i in range(1, 9):
			ref_sum = float(stat[i])

		ref_sum = ref_sum/8
		error = abs((ref_sum - my_sum)/my_sum) 
		if (error >= threshold):
			result.append([stat[0], error]) #file output is [file_name, 0.9]

	result = sorted(result, key = lambda x: x[1], reverse = True)

	for item in result:
		results.write(str(item))
		results.write('\n')

	return

def main():
	measurement = detectFace()
	biometrics = readFile()	

	compare(measurement, biometrics)

	biofile.close()
	results.close()
	return

if __name__ == "__main__":
	main()
