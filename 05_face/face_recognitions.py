import cv2
import face_recognition
import pickle
import time 

image_file = 'D:\Github\Vision_WS\OpenCV_Part1\images\marathon_01.jpg'
encoding_file = 'D:\Github\Vision_WS\OpenCV_Part1/05_face\encodings.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'      # 'hog' or 'cnn'

def detectAndDisplay(image):
  start_time = time.time()
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
  
  # 얼굴 인식 
  boxes = face_recognition.face_locations(rgb, model=model_method)  # 얼굴 위치 찾기
  encodings = face_recognition.face_encodings(rgb, boxes)
  
  names = []  # 이름 초기화
  
  for encoding in encodings:
    # 얼굴 인코딩 비교
    matches = face_recognition.compare_faces(data["encodings"], encoding)  # 얼굴 인코딩 비교
    name = unknown_name  # 이름 초기화
    
    if True in matches:  # 매칭된 경우
      matchedIdxs = [i for (i, b) in enumerate(matches) if b]  # 매칭된 인덱스 추출
      counts = {}  # 딕셔너리 초기화
            
      for i in matchedIdxs:
        name = data["names"][i]  # 이름 설정
        counts[name] = counts.get(name, 0) + 1  # 이름 카운트
      
      name = max(counts, key=counts.get)  # 가장 많이 매칭된 이름 설정
   
    names.append(name)  # 이름 추가

  # 화면에 얼굴 및 이름 표시
  for ((top, right, bottom, left), name) in zip(boxes, names):
    # 얼굴 영역에 사각형 그리기(찾은 경우)
    y = top - 15 if top - 15 > 15 else top + 15  # 이름 표시 위치 계산
    color = (0, 255, 0)       # 색상 설정(초록색)
    line = 2                  # 두께 설정
    
    # 이름이 Unknown인 경우(못 찾은 경우)
    if name == unknown_name:  # 이름이 Unknown인 경우
      color = (0, 0, 255)     # 색상 설정(빨강색)
      line = 1                # 두께 설정
      name = ''               # 이름 초기화
      
    cv2.rectangle(image, (left, top), (right, bottom), color, line)  # 사각형 그리기
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)  # 이름 표시
  
  end_time = time.time()
  process_time = end_time - start_time
  print("=== A frame took {:.3f} seconds".format(process_time))
  # 이미지 출력
  cv2.imshow("Recognition", image)
  
# 피클 파일 로드
data = pickle.loads(open(encoding_file, "rb").read())  # 인코딩 파일 로드

# 이미지 파일 로드
image = cv2.imread(image_file)  # 이미지 파일 로드
detectAndDisplay(image)         # 얼굴 탐지 및 표시

cv2.waitKey(0)                  # 키 이벤트 대기
cv2.destroyAllWindows()         # 모든 윈도우 창 닫기

