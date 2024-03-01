import cv2
import face_recognition
import pickle
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

image_file = 'D:\Github\Vision_WS\OpenCV_Part1\images\marathon_01.jpg'
encoding_file = 'D:\Github\Vision_WS\OpenCV_Part1/05_face\encodings.pickle'
unknown_name = 'Unknown'
title_name = 'Face Recognition'
model_method = 'cnn'    # 'hog' or 'cnn'

##############################################
# 파일을 선택하여 이미지를 읽어오는 함수
##############################################
def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    detectAndDisplay(read_image)

##############################################
# 얼굴 인식 및 이름 표시 함수
##############################################
def detectAndDisplay(image):
  start_time = time.time()                                  # 시작 시간 설정
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # BGR을 RGB로 변환

  # 얼굴 인식
  boxes = face_recognition.face_locations(rgb, model=model_method)  # 얼굴 위치 찾기
  encodings = face_recognition.face_encodings(rgb, boxes)   # 얼굴 인코딩 찾기
  
  names = []  # 이름 초기화
  
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
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    # RGB를 BGR로 변환
  image = Image.fromarray(image)                    # 이미지 배열을 이미지 객체로 변환
  imgtk = ImageTk.PhotoImage(image=image)           # 이미지 객체를 이미지 레이블로 변환
  detection.config(image=imgtk)                     # 이미지 레이블 업데이트
  detection.imgtk = imgtk                           # 이미지 레이블 업데이트
    
# main window
main = Tk()
main.title(title_name)
main.geometry()

# 피클 파일 로드
data = pickle.loads(open(encoding_file, "rb").read())     # 인코딩 파일 로드

# 이미지를 읽어와 BGR를 RGB로 변경 
read_image = cv2.imread(image_file)                       # 이미지 파일 로드
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)       # BGR을 RGB로 변환
image = Image.fromarray(image)                            # 이미지 배열을 이미지 객체로 변환
imgtk = ImageTk.PhotoImage(image=image)                   # 이미지 객체를 이미지 레이블로 변환
(height, width) = read_image.shape[:2]                    # 이미지 높이, 너비 설정

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)
Button(main, text= "파일 선택", height=2,command=lambda:selectFile()).grid(row=1, column=0, columnspan=4)
detection = Label(main, image=imgtk)
detection.grid(row=2, column=0, columnspan=4)

detectAndDisplay(read_image)          # 얼굴 인식 및 이름 표시 함수 호출

main.mainloop()