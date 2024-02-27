import cv2
import numpy as np

min_confidence = 0.5                                           # 최소 신뢰도 설정

# Yolo 로딩   
net = cv2.dnn.readNet("yolo\yolov3-spp.weights", "yolo\yolov3-spp.cfg")         # Yolo 가중치와 설정파일 로딩
classes = []                                                  # 클래스 이름 

with open("yolo\coco.names", "r") as f:                            # 클래스 이름(80개) 로딩 (coco.names 파일에 클래스 이름이 저장되어 있음.
    classes = [line.strip() for line in f.readlines()]        # 클래스 이름 리스트로 설정 

layer_names = net.getLayerNames()                             # Yolo 레이어 이름 로딩
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Yolo 출력 레이어 이름 로딩
print(output_layers)
colors = np.random.uniform(0, 255, size=(len(classes), 3))    # 클래스별 서로 다른 색상 지정

# 이미지 로딩
img = cv2.imread("images/yolo_01.jpg")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)                   # 이미지 크기 조정
height, width, channels = img.shape                           # 이미지 크기 저장
cv2.imshow("Original Image", img)

# 객체 검출(이미지를 Yolo 입력 형식으로 변환 - 3가지 타입)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # 이미지를 Yolo 입력 형식으로 변환

net.setInput(blob)                   # Yolo 입력 설정
outs = net.forward(output_layers)    # Yolo 객체 검출

# 정보를 화면에 표시
class_ids = []                       # 클래스 ID
confidences = []                     # 신뢰도
boxes = []                           # 박스 위치

for out in outs:                     # 객체 검출 정보를 반복
  for detection in out:              # 객체 검출 정보를 반복
    scores = detection[5:]           # 클래스별 신뢰도
    class_id = np.argmax(scores)     # 가장 높은 신뢰도 클래스 ID
    confidence = scores[class_id]    # 가장 높은 신뢰도

    if confidence > min_confidence:             # 신뢰도가 50% 이상인 경우
      # 객체 감지 
      center_x = int(detection[0] * width)      # 객체 중심 x 좌표
      center_y = int(detection[1] * height)     # 객체 중심 y 좌표
      w = int(detection[2] * width)             # 객체 너비
      h = int(detection[3] * height)            # 객체 높이

      # 박스 좌표
      x = int(center_x - w / 2)                 # 객체 좌상단 x 좌표
      y = int(center_y - h / 2)                 # 객체 좌상단 y 좌표

      boxes.append([x, y, w, h])                # 박스 좌표 저장
      confidences.append(float(confidence))     # 신뢰도 저장
      class_ids.append(class_id)                # 클래스 ID 저장

# 비 최대 억제 알고리즘(NMS) 적용
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)  # 노이즈 제거하여 하나의 객체만 검출
print(indexes)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):                    # 박스를 그리기 위한 반복
  if i in indexes:                             # NMS 결과에 있는 경우
    x, y, w, h = boxes[i]                      # 박스 좌표
    label = str(classes[class_ids[i]])         # 클래스 이름
    print(i, label, confidences[i])
    color = colors[i]                          # 색상
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)            # 박스 그리기
    cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)   # 클래스 이름 표시

cv2.imshow("Yolo Image", img)                  # 이미지 출력
cv2.waitKey(0)                                 # 키 대기
cv2.destroyAllWindows()                        # 윈도우 제거
