import numpy as np
import dlib
import cv2
import time
import pygame

pygame.mixer.init()     # mixer 모듈 초기화
# pygame.mixer.music.load('D:/Github/Vision_WS/OpenCV_Part1/music/alram.mp3')     # 음악 파일 로드
pygame.mixer.music.load('D:\Github\Vision_WS\OpenCV_Part1/music/Ring07.wav')     # 음악 파일 로드

RIGHT_EYE = list(range(36, 42))     # 오른쪽 눈의 인덱스
LEFT_EYE = list(range(42, 48))      # 왼쪽 눈의 인덱스
EYES = list(range(36, 48))          # 양쪽 눈의 인덱스
frame_width = 640                   # 비디오 프레임 너비
frame_height = 480                  # 비디오 프레임 높이

title_name = 'Face Drowsiness Detection'
elapsed_time = 0

face_cascade_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()   # 얼굴 검출을 위한 Haar-like 특징 분류기 생성

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):   # Haar-like 특징 분류기 파일 로드 실패 시
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = 'D:/Github/Vision_WS/OpenCV_Part1/model/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)    # 얼굴 랜드마크 검출기 생성

status = 'Awake'        # 초기 상태는 'Awake'
number_closed = 0       # 눈 감은 횟수
min_EAR = 0.25          # 눈 감음 판단을 위한 최소 EAR
closed_limit = 7        # 눈 감음 판단을 위한 최대 횟수
show_frame = None       # 프레임 출력을 위한 변수
sign = None             # 경고 표시를 위한 변수
color = None            # 경고 표시 색상을 위한 변수

def getEAR(points):                                 # EAR 계산 함수
    A = np.linalg.norm(points[1] - points[5])       # 눈의 높이 계산
    B = np.linalg.norm(points[2] - points[4])       # 눈의 높이 계산
    C = np.linalg.norm(points[0] - points[3])       # 눈의 길이 계산                 
    return (A + B) / (2.0 * C)                      # EAR 계산

def detectAndDisplay(image):    # 얼굴 검출 및 랜드마크 검출 함수
    global number_closed        # 눈 감은 횟수            
    global color                # 경고 표시 색상
    global show_frame           # 프레임 출력을 위한 변수
    global sign                 # 경고 표시
    global elapsed_time         # 경과 시간

    start_time = time.time()    # 시작 시간 저장

    #height,width = image.shape[:2]   
    image = cv2.resize(image, (frame_width, frame_height))      # 프레임 크기 조정
    show_frame = image                                          # 프레임 출력을 위한 변수에 프레임 저장
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        # 프레임을 흑백으로 변환
    frame_gray = cv2.equalizeHist(frame_gray)                   # 히스토그램 평활화
    faces = face_cascade.detectMultiScale(frame_gray)           # 얼굴 검출

    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 얼굴 주변에 사각형 그리기

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))   # dlib 사각형 생성

        points = np.matrix([[p.x, p.y] for p in predictor(frame_gray, rect).parts()])   # 얼굴 랜드마크 검출
        show_parts = points[EYES]   # 양쪽 눈의 좌표 추출

        right_eye_EAR = getEAR(points[RIGHT_EYE])           # 오른쪽 눈의 EAR 계산
        left_eye_EAR = getEAR(points[LEFT_EYE])             # 왼쪽 눈의 EAR 계산
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2   # 양쪽 눈의 EAR 평균 계산

        right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")  # 오른쪽 눈 중심 좌표 계산
        left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")    # 왼쪽 눈 중심 좌표 계산

        cv2.putText(image, "{:.2f}".format(right_eye_EAR), (right_eye_center[0,0], right_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 오른쪽 눈 EAR 출력
        cv2.putText(image, "{:.2f}".format(left_eye_EAR), (left_eye_center[0,0], left_eye_center[0,1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 왼쪽 눈 EAR 출력

        for (i, point) in enumerate(show_parts):
            x = point[0,0]                                      # 랜드마크 x 좌표
            y = point[0,1]                                      # 랜드마크 y 좌표
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)     # 눈의 랜드마크 좌표에 점 그리기

        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if( number_closed < 0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed = number_closed + 1

        sign = status + ', Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)
        if( number_closed > closed_limit ):             # 눈 감음 횟수가 최대 횟수를 넘으면
            show_frame = frame_gray                     # 프레임을 흑백으로 변경
            # play SOUND
            if(pygame.mixer.music.get_busy()==False):   # 음악이 재생 중이 아닐 때
                pygame.mixer.music.play()               # 음악 재생

    cv2.putText(show_frame, sign , (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 상태 출력
    cv2.imshow(title_name, show_frame)                  # 프레임 출력
    frame_time = time.time() - start_time               # 한 프레임당 걸린 시간 계산
    elapsed_time += frame_time                          # 경과 시간 계산
    print("Frame time {:.3f} seconds".format(frame_time))

vs = cv2.VideoCapture(0)                    # 비디오 캡처 생성
time.sleep(2.0)                             # 2초 대기
if not vs.isOpened:                         # 비디오 캡처 실패 시
    print('### Error opening video ###')
    exit(0)
while True:                                 # 비디오 프레임 처리 루프
    ret, frame = vs.read()                  # 비디오 프레임 읽기
    if frame is None:                       # 프레임이 없을 경우
        print('### No more frame ###')
        vs.release()                        # 비디오 캡처 해제
        break
    detectAndDisplay(frame)                 # 얼굴 검출 및 랜드마크 검출 함수 호출
    if cv2.waitKey(1) & 0xFF == ord('q'):   # 'q' 키를 누르면 종료
        break


vs.release()                                # 비디오 캡처 해제
cv2.destroyAllWindows()                     # 모든 윈도우 창 닫기