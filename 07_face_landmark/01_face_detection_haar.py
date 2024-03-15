import cv2
import numpy as np

def detectAndDisplay(frame):
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 회색으로 변환
  frame_gray = cv2.equalizeHist(frame_gray)               # 히스토그램 평활화
  
  # 얼굴 검출
  faces = face_cascade.detectMultiScale(frame_gray)       # 얼굴 검출
  
  for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)              # 얼굴 중심 좌표   
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)  # 얼굴 영역에 초록색 사각형 그리기
    faceROI = frame_gray[y:y+h, x:x+w]        # 얼굴 영역 추출
    
    # 각 얼굴에 대해 눈 검출
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2, y2, w2, h2) in eyes:
      eye_center = (x + x2 + w2//2, y + y2 + h2//2)   # 눈 중심 좌표
      radius = int(round((w2 + h2)*0.25))             # 눈 반지름
      frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)  # 눈 중심에 파란색 원 그리기
    cv2.imshow('Capture - Face detection', frame)     # 얼굴 및 눈 검출 결과 화면 출력

# .py를 사용할 경우 전체 경로를 설정해야 오류가 발생하지 않음 
img = cv2.imread('D:\Github\Vision_WS\OpenCV_Part1\images\marathon_02.jpg')
print(f"width: {img.shape[1]} pixels")
print(f"height: {img.shape[0]} pixels")
print(f"channels: {img.shape[2]}")

(height, width) = img.shape[:2]     # 높이, 너비, 채널을 가져옴

cv2.imshow('Image', img)

face_cascade_name = 'D:\Github\Vision_WS\OpenCV_Part1\data\haarcascades\haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'D:\Github\Vision_WS\OpenCV_Part1\data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
  print('--(!)Error loading face cascade')
  exit(0)
  
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
  print('--(!)Error loading eyes cascade')
  exit(0)
  
detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()