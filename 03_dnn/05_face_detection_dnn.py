# 라이브러리 불러오기 
import cv2
import numpy as np

print("OpenCV version:" + cv2.__version__)

# 모델과 메타 정보 가져오기 
model_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/dnn/res10_300x300_ssd_iter_140000.caffemodel'   # 모델 파일
prototxt_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/dnn/deploy.prototxt.txt'                     # prototxt 파일
min_confidence = 0.15   # 최소 신뢰도(임계값 조정)
file_name = "D:/Github/Vision_WS/OpenCV_Part1/images/marathon_01.jpg"   # 파일 이름

#################################################
# 얼굴 탐지 함수 
#################################################
def detectAndDisplay(frame):
  
  # blob을 카페 모델에 전달하고 탐지 결과를 획득
  model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)  # 모델 불러오기
  
  # 이미지 크기 변경하고 정규화 수행 
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
  model.setInput(blob)                            # blob을 모델에 전달
  detections = model.forward()                    # 탐지 결과 획득 
  
  print(detections[0, 0, 1])
  print(detections.shape[2])

  # 탐지된 객체에 대해 반복 수행 
  for i in range(0, detections.shape[2]):
    # 예측과 관련된 신뢰도 추출 
    confidence = detections[0, 0, i, 2]           # 신뢰도 추출
        
    if confidence > min_confidence:               # 최소 신뢰도보다 큰 경우      
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height]) # 탐지된 객체의 경계 상자 추출
      (startX, startY, endX, endY) = box.astype("int")                          # 경계 상자 좌표 추출
      print(f"confidence: {confidence}, startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")
      
      # 연관된 확률과 함께 얼굴의 경계 상자 그리기
      text = f"{confidence * 100:.2f}%"           # 신뢰도 표시 %로 변환
      y = startY - 10 if startY - 10 > 10 else startY + 10  # 신뢰도 표시 위치 계산(맨 위에 표시하지 않도록 처리)
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 경계 상자 그리기
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # 신뢰도 표시
  
  # 이미지 보이기 
  cv2.imshow("Face Detection by DNN", frame)    
#------------------------------ end of detectAndDisplay() ------------------------------#

# 이미지 읽고 기본 정보 출력
img = cv2.imread(file_name)
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]         # 이미지 크기 추출

cv2.imshow("Original Image", img)       # 원본 이미지 보이기

detectAndDisplay(img)                   # 얼굴 탐지 수행

# 키 입력 대기 및 모든 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()