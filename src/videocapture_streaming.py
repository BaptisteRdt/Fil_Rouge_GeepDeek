import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      self.last_time = time.time()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)
      delay = 1 / self.fps - time.time() + self.last_time
      if delay > 0:
        time.sleep(delay)

  def read(self):
    return self.q.get()

# path = 'uav0000013_00000_v/%7d.jpg'
# cap = VideoCapture(path)


# while True:
#   frame = cap.read()

#   # do your model computation

#   cv2.imshow("frame", frame)
#   if chr(cv2.waitKey(1)&255) == 'q':
#     break
