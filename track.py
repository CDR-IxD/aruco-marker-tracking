import numpy as np
import cv2
import json
import websocket
import time
import math
import logging
logging.basicConfig()

# url = "udp://localhost:2000"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

BASE_DIAMETER = 241 # 9.5 inches (24 cm?)

# FLOOR_TO_FIDUCIAL_HEIGHT = 1524 # mm
FLOOR_FIDUCIAL_EDGE_SIZE = 201 # 8 inches (20.3 cm?)
TRACK_FIDUCIAL_EDGE_SIZE = 124 # 4.875 inches (13.7 cm?)
OFFSET_ALONG = -279 #169
OFFSET_ACROSS = -190 #253

floor_scale = 1

# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

ws = None

def dist(x, y):
  return np.linalg.norm(y-x)

def update_ALONG(x):
  global OFFSET_ALONG
  OFFSET_ALONG = -x
def update_ACROSS(x):
  global OFFSET_ACROSS
  OFFSET_ACROSS = -x

camera_matrix = np.array([
  [857.4829, 0.0, 968.0622],
  [0.0, 876.718, 556.3714],
  [0.0, 0.0, 1.0]])
dist_coeffs = np.array([[-2.57614e-01, 8.77087e-02, -2.569708e-04, -5.933904e-04, -1.5219409e-02]])

def run():
  cv2.namedWindow('frame')

  cv2.createTrackbar('X', 'frame', OFFSET_ALONG, 1000, update_ALONG)
  cv2.createTrackbar('Y', 'frame', OFFSET_ACROSS, 1000, update_ACROSS)

  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    print (frame.shape)
    h, w, _ = frame.shape

    # Our operations on the frame come here
    gray_d = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray_d # cv2.undistort(gray_d, camera_matrix, dist_coeffs)
    
    res = cv2.aruco.detectMarkers(gray,dictionary)
    # print(res[0],res[1],len(res[2]))
    
    if len(res[0]) > 0:
      cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
      data = {
        'updates': [],
        'size': { 'width': w, 'height': h } # for aspect ratio calculation
      }

      # print(res[0], res[1])

      for (fids, index) in zip(res[0], res[1]):
        fid = fids[0]
        if int(index[0]) == 0: # floor fiducial!
          # print("found floor!")
          d = dist(fid[0], fid[1])
          global floor_scale
          floor_scale = d / FLOOR_FIDUCIAL_EDGE_SIZE
        else:
          # print("found non-floor!")
          center = sum(fid)/4.0
          front = (fid[0]+fid[1])/2.0
          data['updates'].append({
            'id': int(index[0]), 
            'fiducialLocation': [[int(pt[0]), int(pt[1])] for pt in fid],
            'fiducialScale': dist(fid[0], fid[1]) / TRACK_FIDUCIAL_EDGE_SIZE,
            'angle': math.atan2(front[1]-center[1], front[0]-center[0])
          })
      for update in data['updates']:
        Sf = update['fiducialScale']
        Sr = floor_scale
        angle = update['angle']
        
        x_factor = OFFSET_ALONG * math.cos(angle) - OFFSET_ACROSS * math.sin(angle)
        y_factor = OFFSET_ALONG * math.sin(angle) + OFFSET_ACROSS * math.cos(angle)
        
        robotLocation = [(
          int((float(pt[0]-w/2) / Sf + x_factor) * Sr + w/2), 
          int((float(pt[1]-h/2) / Sf + y_factor) * Sr + h/2)
        ) for pt in update['fiducialLocation'] ]
                                    
        update['robotLocation'] = robotLocation
        update['location'] = [{ 'x': float(pt[0])/w, 'y': float(pt[1])/h } for pt in robotLocation]

        for i in [(1,2), (2,3), (3,0)]:
          cv2.line(gray, robotLocation[i[0]], robotLocation[i[1]], 255, 3)

      # print(data)
      if len(data['updates']) > 0 and ws is not None:
        ws.send(json.dumps(data))

    # Display the resulting frame
    cv2.imshow('frame',gray)
    print("shown!")
    if cv2.waitKey(1) & 0xFF == ord('q'):
      # When everything done, release the capture
      cap.release()
      cv2.destroyAllWindows()
      import sys
      sys.exit()
  

def on_message(ws, message):
  pass
  # print(message)

def on_error(ws, error):
  print(error)

def on_close(ws):
  print("### closed ###")
  print("### trying to reconnect ###")
  time.sleep(3)
  start()

def on_open(ws):
  print("### connected ###")
  run()

def start():
  # websocket.enableTrace(True)
  global ws
  ws = websocket.WebSocketApp("ws://localhost:5000/bot-updates",
                            on_message = on_message,
                            on_error = on_error,
                            on_close = on_close)
  ws.on_open = on_open
  ws.run_forever()

if __name__ == "__main__":
  # this might work?
  start()
  # run()
