import cv2
import numpy as np

# 파일명일 때 이름 조심해서 사용( \r 같은 문자가 있으면 안됨)
model_name = "D:/Github/Vision_WS/OpenCV_Part1/data/dnn/res10_300x300_ssd_iter_140000.caffemodel"  # 모델 파일
prototxt_name = "D:\Github\Vision_WS\OpenCV_Part1\data\dnn\deploy.prototxt.txt"  # prototxt 파일
min_confidence = 0.5  # 최소 신뢰도(임계값 조정)
file_name = "D:/Github/Vision_WS/OpenCV_Part1/videos/tedy_01.mp4"  # 파일 이름

def detectAndDisplay(frame):
  # blob을 모델에 전달하고 탐지 결과를 획득
  model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)  # 모델 불러오기

  # 이미지 크기 변경하고 정규화 수행
  blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
  )
  model.setInput(blob)          # blob을 모델에 전달
  detections = model.forward()  # 탐지 결과 획득

  # 탐지된 객체에 대해 반복 수행
  for i in range(0, detections.shape[2]):
    # 예측과 관련된 신뢰도 추출
    confidence = detections[0, 0, i, 2]  # 신뢰도 추출

    if confidence > min_confidence:  # 최소 신뢰도보다 큰 경우만 처리 
      (height, width) = frame.shape[:2]      
      # 탐지된 객체의 경계 상자 추출
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])  
      (startX, startY, endX, endY) = box.astype("int")  # 경계 상자 좌표 추출
      print(f"confidence: {confidence}, startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")

      # 연관된 확률과 함께 얼굴의 경계 상자 그리기
      text = f"{confidence * 100:.2f}%"
      y = startY - 10 if startY - 10 > 10 else startY + 10  # 신뢰도 표시 위치 계산(맨 위에 표시하지 않도록 처리)
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      
  # 이미지 표시
  cv2.imshow("Face Detection by dnn", frame)

# 비디오 스트림 읽기
cap = cv2.VideoCapture(file_name)
if not cap.isOpened:                            # 비디오 스트림 열기에 실패한 경우
  print("--(!)Error opening video capture")
  exit(0)
  
while True:                                     # 비디오 스트림 읽기
  ret, frame = cap.read()                       # 비디오 스트림에서 프레임 읽기    
  if frame is None:                             # 프레임을 읽지 못한 경우
    print("--(!) No captured frame -- Break!")
    break
  
  detectAndDisplay(frame)                       # 탐지 및 표시 함수 호출
  
  # 'q'를 눌러 종료하기 
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cv2.destroyAllWindows()                         # 모든 창 닫기
