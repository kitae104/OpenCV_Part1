import cv2
import face_recognition
import pickle

dataset_paths = ['D:\Github\Vision_WS\OpenCV_Part1\data\son/', 
                 'D:\Github\Vision_WS\OpenCV_Part1\data/tedy/'] # 데이터셋 경로
names = ['Son', 'Tedy']                           # 이름
number_images = 10                                # 이미지 개수
image_type = '.jpg'                               # 이미지 타입
encoding_file = 'encodings.pickle'                # 인코딩 파일

# cnn or hog (dlib 기반 딥러닝 얼굴 탐지기 모델)
model_method = 'cnn'                              # 모델 방식

knownEncodings = []                               # 인코딩 초기화
knownNames = []                                   # 이름 초기화

for (i, dataset_paths) in enumerate(dataset_paths):  
  name = names[i]                                             # 이름 설정 
  
  for idx in range(number_images):                            # 이미지 개수만큼 반복
    file_name = dataset_paths + str(idx + 1) + image_type     # 파일 이름 설정
    image = cv2.imread(file_name)                             # 이미지 파일 로드
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # BGR을 RGB로 변환  

    boxes = face_recognition.face_locations(image, model=model_method)
    encodings = face_recognition.face_encodings(image, boxes)

    # 인코딩된 얼굴 추가
    for encoding in encodings:
      print(f'file_name : {file_name}, name : {name}, encoding : {encoding}')
      knownEncodings.append(encoding)
      knownNames.append(name)
      
# 인코딩된 얼굴과 이름을 디스크에 저장
data = {"encodings": knownEncodings, "names": knownNames}   # 데이터 설정
f = open(encoding_file, "wb")         # 파일 열기
f.write(pickle.dumps(data))           # 파일에 데이터 쓰기
f.close()                             # 파일 닫기