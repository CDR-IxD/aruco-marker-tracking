import numpy as np
import cv2
import json
import websocket
import thread
import time

url = "udp://localhost:2000"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)

# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

ws = None

def run():
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print (frame.shape)
    h, w, _ = frame.shape

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.aruco.detectMarkers(gray,dictionary)
    # print(res[0],res[1],len(res[2]))
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
      # print(data)
      ws.send(json.dumps(data))

    if len(res[0]) > 0:
      cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      # When everything done, release the capture
      cap.release()
      cv2.destroyAllWindows()
      quit()
  

def on_message(ws, message):
  print(message)

def on_error(ws, error):
  print(error)

def on_close(ws):
  print("### closed ###")
  print("### trying to reconnect ###")
  time.sleep(3)
  start()

def on_open(ws):
  run()

def start():
  websocket.enableTrace(True)
  global ws
  ws = websocket.WebSocketApp("ws://localhost:5000/bot-updates",
                            on_message = on_message,
                            on_error = on_error,
                            on_close = on_close)
  ws.on_open = on_open
  ws.run_forever()

if __name__ == "__main__":
  start()