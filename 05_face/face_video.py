import cv2
import face_recognition
import pickle
import time

file_name = 'videos/son_02.mp4'
encoding_file = 'D:\Github\Vision_WS\OpenCV_Part1/05_face\encodings.pickle'
unknown_name = 'Unknown'
model_method = 'hog'                                      # 'hog' or 'cnn'  
output_name = 'videos/output_' + model_method + '.avi'

##############################################
# 얼굴 인식 및 이름 표시 함수
##############################################
def detectAndDisplay(image):
  start_time = time.time()  # 시작 시간 저장
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
  
  # 얼굴 인식
  boxes = face_recognition.face_locations(rgb, model=model_method)  # 얼굴 위치 찾기
  encodings = face_recognition.face_encodings(rgb, boxes)           # 얼굴 인코딩 찾기
  
  names = []  # 이름 초기화

  # 인코딩 내용으로 처리 
  for encoding in encodings:
    # 얼굴 인코딩 비교
    matches = face_recognition.compare_faces(data["encodings"], encoding)  # 얼굴 인코딩 비교 
    name = unknown_name   # 이름 초기화

    # 매칭된 경우
    if True in matches:
      # 매칭된 인덱스 추출한 후 얼굴이 매칭된 횟수를 카운트
      matchedIdxs = [i for (i, b) in enumerate(matches) if b]
      counts = {}

      # 일치하는 인덱스의 이름을 추출하고 카운트 증가
      for i in matchedIdxs:
        name = data["names"][i]
        counts[name] = counts.get(name, 0) + 1

      name = max(counts, key=counts.get)    # 가장 많이 매칭된 이름 설정
      
    names.append(name)  # 이름 리스트에 이름 추가
  
  # 화면에 얼굴 및 이름 표시
  for ((top, right, bottom, left), name) in zip(boxes, names):
  # 얼굴 영역에 사각형 그리기(찾은 경우)
    y = top - 15 if top - 15 > 15 else top + 15   # 이름 표시 위치 계산
    color = (0, 255, 0)                           # 색상 설정(초록색)
    line = 2                                      # 두께 설정
    if(name == unknown_name):                     # 이름이 Unknown인 경우(못 찾은 경우) 
        color = (0, 0, 255)                       # 색상 설정(빨강색)
        line = 1                                  # 두께 설정
        name = ''                                 # 이름 초기화
        
    cv2.rectangle(image, (left, top), (right, bottom), color, line) # 사각형 그리기
    y = top - 15 if top - 15 > 15 else top + 15   # 이름 표시 위치 계산
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line) # 이름 표시
    
  end_time = time.time()                            # 종료 시간 설정    
  process_time = end_time - start_time              # 경과 시간 계산
  print(f"=== A frame took {process_time:.3f} seconds")

  # 이미지 출력
  image = cv2.resize(image, None, fx=0.5, fy=0.5)  # 이미지 크기 조정
  cv2.imshow("Recognition", image)                 # 이미지 출력
  
  global writer                                    # 글로벌 객체 사용 
  if writer is None and output_name is not None:   # 비디오 작성 객체가 없는 경우
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")       # 코덱 설정
    writer = cv2.VideoWriter(output_name, fourcc, 24, (image.shape[1], image.shape[0]), True)  # 비디오 작성 객체 설정
  
  if writer is not None:                           # 비디오 작성 객체가 있는 경우
    writer.write(image)                           # 비디오 작성
  
# 피클 파일 로드
data = pickle.loads(open(encoding_file, "rb").read())     # 인코딩 파일 로드

# 동영상 파일 열기
cap = cv2.VideoCapture(file_name)  # 동영상 파일 열기
writer = None

if not cap.isOpened:
  print('--(!)Error opening video capture')
  exit(0)
  
while True:
  ret, frame = cap.read()
  if frame is None:
    print('--(!) No captured frame -- Break!')     
    cap.release()                   # 비디오 파일 닫기     
    writer.release()                # 비디오 작성 객체 닫기
    break
  
  detectAndDisplay(frame)           # 얼굴 인식 및 표시
  
  if cv2.waitKey(10) == 27:         # ESC 키를 누르면 종료
    break

cv2.destroyAllWindows()      # 모든 윈도우 창 닫기