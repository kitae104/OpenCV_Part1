import numpy as np
import cv2
import time

face_model = 'D:\Github\Vision_WS\OpenCV_Part1/model/dnn/res10_300x300_ssd_iter_140000.caffemodel'                    # 얼굴 검출 모델
face_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1/model/dnn/deploy.prototxt.txt'                                      # 얼굴 검출 모델(메타 정보)
age_model = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/age_net.caffemodel'             # 나이 검출 모델
age_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/deploy_age.prototxt'         # 나이 검출 모델(메타 정보)
gender_model = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/gender_net.caffemodel'       # 성별 검출 모델
gender_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/deploy_gender.prototxt'   # 성별 검출 모델(메타 정보)

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] # 나이 리스트 
gender_list = ['Male','Female'] # 성별 리스트

title_name = 'Age and Gender Recognition'   # 윈도우 타이틀 이름
min_confidence = 0.5                        # 최소 확률 값  
recognition_count = 0                       # 인식 카운트
elapsed_time = 0                            # 경과 시간
OUTPUT_SIZE = (300, 300)                    # 출력 크기

# 모델 로딩 및 설정
detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)              # 얼굴 검출 모델 로딩
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)            # 나이 검출 모델 로딩
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)   # 성별 검출 모델 로딩

########################################
# 얼굴 인식 함수
########################################
def detectAndDisplay(image):
    start_time = time.time()        # 시작 시간      
    (h, w) = image.shape[:2]        # 이미지 크기

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, OUTPUT_SIZE), 1.0, OUTPUT_SIZE,
        (104.0, 177.0, 123.0), swapRB=False, crop=False)    # 이미지 전처리
    
    detector.setInput(imageBlob)        # 얼굴 검출 모델에 이미지 설정
    detections = detector.forward()     # 얼굴 검출

    for i in range(0, detections.shape[2]):     # 검출된 얼굴 수 만큼 반복
        confidence = detections[0, 0, i, 2]     # 확률 추출

        if confidence > min_confidence:         # 최소 확률보다 큰 경우
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # 박스 좌표 계산
            (startX, startY, endX, endY) = box.astype("int")        # 박스 좌표

            face = image[startY:endY, startX:endX]  # 얼굴 영역 추출
            (fH, fW) = face.shape[:2]               # 얼굴 영역 크기

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),swapRB=False) # 얼굴 영역 전처리

            # 나이 인식    
            age_detector.setInput(face_blob)                # 나이 검출 모델에 이미지 설정
            age_predictions = age_detector.forward()        # 나이 검출
            age_index = age_predictions[0].argmax()         # 나이 인덱스
            age = age_list[age_index]                       # 나이
            age_confidence = age_predictions[0][age_index]  # 나이 확률
            
            # 성별 인식
            gender_detector.setInput(face_blob)             # 성별 검출 모델에 이미지 설정
            gender_predictions = gender_detector.forward()  # 성별 검출
            gender_index = gender_predictions[0].argmax()   # 성별 인덱스
            gender = gender_list[gender_index]              # 성별
            gender_confidence = gender_predictions[0][gender_index] # 성별 확률

            text = "{}: {:.2f}% {}: {:.2f}%".format(gender, gender_confidence*100, age, age_confidence*100) # 텍스트 생성
            y = startY - 10 if startY - 10 > 10 else startY + 10    # 텍스트 위치
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2) # 박스 그리기
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2) # 텍스트 출력
            
            print('==============================')
            print("Gender {} time {:.2f} %".format(gender, gender_confidence*100))
            print("Age {} time {:.2f} %".format(age, age_confidence*100))
            print("Age     Probability(%)")
            for i in range(len(age_list)):
                print("{}  {:.2f}%".format(age_list[i], age_predictions[0][i]*100))
                
            print("Gender  Probability(%)")
            for i in range(len(gender_list)):
                print("{}  {:.2f} %".format(gender_list[i], gender_predictions[0][i]*100))
                
    frame_time = time.time() - start_time   # 경과 시간 계산
    global elapsed_time                     # 전역 변수 사용
    elapsed_time += frame_time              # 경과 시간 누적
    print("Frame time {:.3f} seconds".format(frame_time))   # 경과 시간 출력
    
    # 이미지 출력
    cv2.imshow(title_name, image)           # 이미지 출력

########################################
# MAIN 부분 
########################################
vs = cv2.VideoCapture(0)                    # 비디오 시작
time.sleep(2.0)                             # 2초 대기

if not vs.isOpened:                         # 비디오 시작 확인
    print('Error Open Camera')
    exit(0)

while True:
    ret, frame = vs.read()                  # 비디오에서 프레임 읽기
    if frame is None:                       # 프레임 읽기 확인
        print('No more frame')
        vs.release()                        # 비디오 종료
        break
    
    detectAndDisplay(frame)                 # 얼굴 인식 함수

    if cv2.waitKey(1) & 0xFF == ord('q'):   # 'q' 키를 누르면 종료
        break

vs.release()                                # 비디오 종료
cv2.destroyAllWindows()                     # 윈도우 제거