import numpy as np
import cv2
import asyncio
import json
import websockets

url = "udp://localhost:2000"
cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

async def run():
  async with websockets.connect('ws://localhost:5000/bot-updates') as ws:
    while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()
      print (frame.shape)
      h, w, _ = frame.shape

      # Our operations on the frame come here
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      res = cv2.aruco.detectMarkers(gray,dictionary)
      print(res[0],res[1],len(res[2]))
      if len(res[0]) > 0:
        data = {
          'updates': 
            [
              {
                'id': int(index[0]), 
                'location': [{'x': float(pt[0]/w), 'y': float(pt[1]/h)} for pt in fid]
              } for (fid, index) in zip(res[0][0], res[1])
            ],
          'size': { 'width': w, 'height': h } # for aspect ratio calculation
        }
        print(data)
        await ws.send(json.dumps(data))

      if len(res[0]) > 0:
        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
      # Display the resulting frame
      cv2.imshow('frame',gray)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
asyncio.get_event_loop().run_until_complete(run())

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()