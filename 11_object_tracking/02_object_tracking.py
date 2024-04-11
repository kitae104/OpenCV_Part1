from tracking import Tracker, Trackable
import cv2
import numpy as np
import time

frame_size = 416          # 프레임 사이즈
frame_count = 0           # 프레임 수
min_confidence = 0.5      # 최소 신뢰도
min_directions = 10       # 최소 방향(한 방향으로 가는 횟수)

height = 0                # 높이
width = 0                 # 너비

count_limit = 0           # 카운트 제한
up_count = 0              # 위로 카운트
down_count = 0            # 아래로 카운트
direction = ''            # 방향

trackers = []             # 트래커
trackables = {}           # 트랙어블

file_name = 'E:/Github/Vision_WS/OpenCV_Part1/videos/Messi.mp4'
output_name = 'E:/Github/Vision_WS/OpenCV_Part1/videos/output_01.avi'

# Load Yolo
net = cv2.dnn.readNet("E:/Github/Vision_WS/OpenCV_Part1/yolo/yolov3-spp.weights", "E:/Github/Vision_WS/OpenCV_Part1/yolo/yolov3.cfg")
layer_names = net.getLayerNames()
# print(layer_names)

output_layers = net.getUnconnectedOutLayers()
print(output_layers)

if len(output_layers) == 0:
    # 빈 리스트 처리 코드
    print("네트워크 출력 레이어가 존재하지 않습니다.")
    # ...
else:
  output_layers = [layer_names[i[0] - 1] for i in output_layers]
  
print(output_layers)


# initialize Tracker 
tracker = Tracker()

# initialize the video writer 
writer = None

def writeFrame(img):
    # use global variable, writer
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (img.shape[1], img.shape[0]), True)

    if writer is not None:
        writer.write(img)  


vs = cv2.VideoCapture(file_name)

# loop over the frames from the video stream
while True:
        ret, frame = vs.read()
        
        if frame is None:
            print('### No more frame ###')
            break
        
        # Start time capture
        start_time = time.time()
        frame_count += 1

        (height, width) = frame.shape[:2]
        count_limit = height // 2
        
        # draw a horizontal line in the center of the frame
        cv2.line(frame, (0, count_limit), (width, count_limit), (0, 255, 255), 2)
        
        # construct a blob for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (frame_size, frame_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        rects = []

        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)                        # 확률이 높은 클래스 ID
                confidence = scores[class_id]                       # 확률
                # Filter only 'person'
                if class_id == 0 and confidence > min_confidence:   # 사람만 추출

                    # Object detected
                    center_x = int(detection[0] * width)            # 중심 x
                    center_y = int(detection[1] * height)           # 중심 y
                    w = int(detection[2] * width)                   # 너비
                    h = int(detection[3] * height)                  # 높이  

                    # Rectangle coordinates
                    x = int(center_x - w / 2)                       # x 좌표 
                    y = int(center_y - h / 2)                       # y 좌표

                    boxes.append([x, y, w, h])                      # 박스
                    
                    confidences.append(float(confidence))           # 신뢰도

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4) # 비 최대 억제
        for i in range(len(boxes)):                                 # 박스 그리기
            if i in indexes:
                x, y, w, h = boxes[i]                               # 박스 좌표
                rects.append([x, y, x+w, y+h])                      # 박스 좌표
                label = '{:,.2%}'.format(confidences[i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)                            # 사각형 그리기
                cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)   # 텍스트 그리기
    
        # use Tracker
        objects = tracker.update(rects)                             # 객체 업데이트

        # loop over the trackable objects
        for (objectID, centroid) in objects.items():                # 객체 ID, 중심 좌표
                # check if a trackable object exists with the object ID
                trackable = trackables.get(objectID, None)          # 트랙어블 객체

                if trackable is None:
                        trackable = Trackable(objectID, centroid)   # 트랙어블 객체 생성
                else:
                        y = [c[1] for c in trackable.centroids]     # 중심 y 좌표
                        variation = centroid[1] - np.mean(y)        # 중심 y 좌표 변화량
                        trackable.centroids.append(centroid)        # 중심 좌표 추가
                        if variation < 0:
                            direction = 1                           # 방향
                        else: 
                            direction = 0                           # 방향
                        trackable.directions.append(direction)      # 방향 추가
                        mean_directions = int(round(np.mean(trackable.directions)))   # 평균 방향
                        len_directions = len(trackable.directions)  # 방향 길이

                        # check to see if the object has been counted or not
                        if (not trackable.counted) and (len_directions > min_directions):   # 카운트
                                if direction == 1 and centroid[1] < count_limit:        # 방향이 위로
                                        up_count += 1               # 위로 카운트
                                        trackable.counted = True    # 카운트
                                elif direction == 0 and centroid[1] > count_limit:      # 방향이 아래로
                                        down_count += 1             # 아래로 카운트
                                        trackable.counted = True    # 카운트  

                # store the trackable object in our dictionary
                trackables[objectID] = trackable                    # 트랙어블 객체 저장
                text = "ID {}".format(objectID)                     # 텍스트
                cv2.putText(frame, text, (centroid[0] + 10, centroid[1] + 10),                              
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)                # 텍스트 그리기
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)     # 원 그리기

        info = [
            ("Up", up_count),
            ("Down", down_count),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):                         # 정보 그리기
            text = "{}: {}".format(k, v)                            # 텍스트
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 텍스트 그리기

        writeFrame(frame)                                           # 프레임 저장
        
        # show the output frame
        cv2.imshow("Frame", frame)                                  # 프레임 출력
        frame_time = time.time() - start_time                       # 프레임 시간
        print("Frame {} time {}".format(frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF                                 # 키 입력    

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
            
vs.release()
writer.release()
cv2.destroyAllWindows()