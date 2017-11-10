import cv2
import numpy as np

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

i = 0
def next():
  global i
  img = cv2.aruco.drawMarker(d, i, 1000)
  cv2.imshow("marker", img)
  cv2.imwrite("marker-"+str(i)+".jpg", img)
  cv2.waitKey(0)
  i = i + 1

while True:
  next()
  
# cv2.imshow("marker0", cv2.aruco.drawMarker(d,i,1000))
# cv2.aruco.drawMarker(Dictionary,10,250,markerImage,1)
# cv2.aruco.drawMarker(Dictionary,20,250,markerImage,1)
# cv2.aruco.drawMarker(Dictionary,25,250,markerImage,1)
# cv2.aruco.drawMarker(Dictionary,50,250,markerImage,1)
# cv2.aruco.drawMarker(Dictionary,100,250,markerImage,1)
# cv2.aruco.drawMarker(Dictionary,200,250,markerImage,1)

# cv2.imshow("markers",markerImage)
# cv2.waitKey(0)

# cv2.imwrite("marker.jpg",markerImage)