####################################################
# 기울어진 얼굴을 정면으로 만들기
# 눈의 중심을 찾고 회전 행렬을 구한 후 회전시킴
####################################################
# 필요한 패키지를 불러옵니다
import numpy as np
import dlib
import cv2

# 눈을 구분하기 위해 상수 준비
RIGHT_EYE = list(range(36, 42))     # 오른쪽 눈
LEFT_EYE = list(range(42, 48))      # 왼쪽 눈
EYES = list(range(36, 48))          # 양쪽 눈 

# 파일 경로
predictor_file = 'D:/Github/Vision_WS/OpenCV_Part1/model/shape_predictor_68_face_landmarks.dat' # 68개 랜드마크 파일
image_file = 'D:/Github/Vision_WS/OpenCV_Part1/images/face.jpg'
MARGIN_RATIO = 1.5                  # 얼굴을 찾을 영역 확대 비율
OUTPUT_SIZE = (300, 300)            # 결과 이미지 크기

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()         # 얼굴 검출기
predictor = dlib.shape_predictor(predictor_file)    # 얼굴 랜드마크 검출기

# 이미지 읽기
image = cv2.imread(image_file)      # 이미지 읽기
image_origin = image.copy()         # 원본 이미지 복사

# 이미지 크기 및 색 조절
(image_height, image_width) = image.shape[:2]       # 이미지 높이, 너비
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 흑백 이미지로 변환 -> 하나의 채널만 사용 

# 얼굴 검출
rects = detector(gray, 1)           # 얼굴 검출
print(f"Number of faces detected : {len(rects)}")   # 검출된 얼굴 수 출력
print(rects)

# 얼굴의 크기를 구하는 함수
def getFaceDimension(rect):
    # 얼굴의 크기를 구하는 함수
    x = rect.left()                     # 왼쪽 x 좌표
    y = rect.top()                      # 위 y 좌표
    w = rect.right() - x                # 얼굴의 너비
    h = rect.bottom() - y               # 얼굴의 높이
    return (x, y, w, h)

# 얼굴을 중심으로 이미지를 크롭할 크기를 구하는 함수
def getCropDimension(rect, center):
    
    width = (rect.right() - rect.left())    # 얼굴의 너비
    half_width = width // 2                 # 얼굴의 너비의 반
    (centerX, centerY) = center             # 얼굴의 중심 좌표
    startX = centerX - half_width           # 얼굴의 중심 좌표에서 얼굴의 반만큼 왼쪽
    endX = centerX + half_width             # 얼굴의 중심 좌표에서 얼굴의 반만큼 오른쪽
    startY = rect.top()                     # 얼굴의 위 y 좌표
    endY = rect.bottom()                    # 얼굴의 아래 y 좌표
    return (startX, endX, startY, endY)     # 얼굴을 크롭할 좌표를 반환

# 검출된 얼굴 개수만큼 반복
for (i, rect) in enumerate(rects):
    (x, y, w, h) = getFaceDimension(rect)                           # 얼굴의 크기를 구함
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)    # 얼굴에 사각형 표시

    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])  # 랜드마크 점들의 좌표
    show_parts = points[EYES]                # 눈만 표시

    # 눈의 중심을 구함
    right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")    # 오른쪽 눈 중심
    left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")      # 왼쪽 눈 중심
    print(f"Right Eye Center : {right_eye_center}, Left Eye Center : {left_eye_center}")    

    # 눈 중심을 표시
    cv2.circle(image, (right_eye_center[0, 0], right_eye_center[0, 1]), 5, (0, 0, 255), -1)   # 오른쪽 눈 중심 표시
    cv2.circle(image, (left_eye_center[0, 0], left_eye_center[0, 1]), 5, (0, 0, 255), -1)     # 왼쪽 눈 중심 표시

    cv2.circle(image, (left_eye_center[0, 0], right_eye_center[0, 1]), 1, (0, 255, 255), -1) # 오른쪽 눈 중심 표시

    # 눈 중심으로 선 그리기
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]), (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 2)
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]), (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1)
    cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]), (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1)

    # 눈 중심을 기준으로 각도를 구함
    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]          # 밑변 구하기
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]          # 높이 구하기
    degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180      # 각도 구하기
    print(f"Degree : {degree}")

    # 이동시 스케일 구하기 
    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))     # 눈 사이의 거리 구하기
    aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0] # 정렬된 눈 사이의 거리 구하기
    scale = aligned_eye_distance / eye_distance                         # 스케일 구하기(이동후 비율)    
    print(f"Scale : {scale}")

    # 눈 중심 점 찾고 파란색 점 찍기
    eyes_center = ((left_eye_center[0,0] + right_eye_center[0,0]) // 2, (left_eye_center[0,1] + right_eye_center[0,1]) // 2)
    cv2.circle(image, eyes_center, 5, (255, 0, 0), -1)    
    print(f"Eyes Center : {eyes_center}, Type : {type(eyes_center)}")
    eyes_center = (int(eyes_center[0]), int(eyes_center[1]))

    # 회전을 위한 행렬 구하기
    metrix = cv2.getRotationMatrix2D(eyes_center , degree, scale)   # 회전 행렬 구하기
    cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0,0], right_eye_center[0,1] + 20),
     	 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)             # 각도 표시

    warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height), flags=cv2.INTER_CUBIC) # 회전 이미지
    
    cv2.imshow("warpAffine", warped)

    # 회전된 이미지 크롭
    (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)  # 얼굴 중심을 기준으로 크롭할 크기 구함
    croped = warped[startY:endY, startX:endX]   # 이미지 크롭(순서 확인 필요 - y, x 순서로 크롭해야 함)
    output = cv2.resize(croped, OUTPUT_SIZE)    # 결과 이미지 OUTPUT_SIZE 크기로 변환
    cv2.imshow("output", output)

    # 결과 이미지 저장
    output_file = 'D:/Github/Vision_WS/OpenCV_Part1/outputs/faces/1_out.jpg'
    cv2.imwrite(output_file, output)            # 결과 이미지 저장

    # 눈 부분에 점 표시 
    for (i, point) in enumerate(show_parts):    # 랜드마크 점 반복
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)  # 눈에 노란색 점 표시

# 결과 이미지 보이기 
cv2.imshow("Face Detection", image)     # 얼굴 검출 결과 출력
cv2.waitKey(0)
cv2.destroyAllWindows()