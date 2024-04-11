import cv2
import time

file_name = 'E:/Github/Vision_WS/OpenCV_Part1/videos/son_01.mp4'  # 비디오 파일 경로
frame_count = 0                                                   # 프레임 수

# csrt
# tracker = cv2.TrackerCSRT_create()        # CSRT 객체 생성

# kcf
tracker = cv2.TrackerKCF_create()           # KCF 객체 생성

# boosting
# tracker = cv2.TrackerBoosting_create()    # Boosting 객체 생성

# mil
# tracker = cv2.TrackerMIL_create()         # MIL 객체 생성

# tld
# tracker = cv2.TrackerTLD_create()         

# medianflow
# tracker = cv2.TrackerMedianFlow_create()  # MedianFlow 객체 생성

# mosse
# tracker = cv2.TrackerMOSSE_create()       # MOSSE 객체 생성

face_cascade_name = 'E:/Github/Vision_WS/OpenCV_Part1/data/haarcascades/haarcascade_frontalface_alt.xml'  # 얼굴 검출 모델
face_cascade = cv2.CascadeClassifier()      # 얼굴 검출기 생성 

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # 얼굴 검출기 로드 
  print('### Error loading face cascade ###') 
  exit(0)                                                         # 종료                    

detected = False                        # 검출 여부
frame_mode = 'Tracking'                 # 프레임 모드
elapsed_time = 0                        # 경과 시간
trackers = cv2.legacy.MultiTracker_create()           # 멀티 트래커 생성

vs = cv2.VideoCapture(file_name)        # 비디오 파일 로드

while True:
  ret, frame = vs.read()                # 프레임 읽기
  if frame is None:                     # 프레임이 없으면 종료
    print('### No more frame ###')    
    break                             
  start_time = time.time()              # 시간 측정
  frame_count += 1                # 프레임 수 증가    
  if detected:
    frame_mode = 'Tracking'
    (success, boxes) = trackers.update(frame)
    for box in boxes:
      (x, y, w, h) = [int(v) for v in box]
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  else:
    frame_mode = 'Detection'
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)            
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

    trackers.add(tracker, frame, tuple(faces[0]))
    detected = True

  cv2.imshow("Frame", frame)
  frame_time = time.time() - start_time
  elapsed_time += frame_time
  print("[{}] Frame {} time {}".format(frame_mode, frame_count, frame_time))
  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
      break

print("Elapsed time {}".format(elapsed_time))
vs.release()
cv2.destroyAllWindows()