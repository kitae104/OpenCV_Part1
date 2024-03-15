# 라이브러리 추가 
import numpy as np
import dlib
import cv2

# 68개의 점을 구분하기 위해 상수 준비 
RIGHT_EYE = list(range(36, 42))     # 오른쪽 눈
LEFT_EYE = list(range(42, 48))      # 왼쪽 눈
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'D:\Github\Vision_WS\OpenCV_Part1\model\shape_predictor_68_face_landmarks.dat'

image_file = 'D:\Github\Vision_WS\OpenCV_Part1\images\\face.jpg'

detector = dlib.get_frontal_face_detector()
predicator = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)        # 얼굴 검출
print(f"Number of faces detected : {len(rects)}")   # 검출된 얼굴 수 출력
print(rects)

for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predicator(gray, rect).parts()])
    show_parts = points[ALL]
    print(show_parts)           # 점들의 좌표 

    for (i, point) in enumerate(show_parts):
        x = point[0, 0]
        y = point[0, 1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, f"{i + 1}", (x, y-2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
cv2.imshow("Face Landmark", image)
cv2.waitKey(0)