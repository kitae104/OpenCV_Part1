import cv2
import numpy as np

from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# 파일명일 때 이름 조심해서 사용( \r 같은 문자가 있으면 안됨) 
model_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/dnn/res10_300x300_ssd_iter_140000.caffemodel'   # 모델 파일
prototxt_name = 'D:\Github\Vision_WS\OpenCV_Part1\data\dnn\deploy.prototxt.txt'                     # prototxt 파일
min_confidence = 0.3   # 최소 신뢰도(임계값 조정)
file_name = "D:\Github\Vision_WS\OpenCV_Part1\images\marathon_01.jpg"   # 파일 이름

title_name = 'dnn Deep Learnig object detection'
frame_width = 300
frame_height = 300

#################################################################################
def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    
    detectAndDisplay(read_image, width, height)
#################################################################################

#################################################################################
def detectAndDisplay(frame, width, height):
  # blob을 모델에 전달하고 탐지 결과를 획득
  model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)  # 모델 불러오기
  
  # 이미지 크기 변경하고 정규화 수행 
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
  model.setInput(blob)                            # blob을 모델에 전달
  detections = model.forward()                    # 탐지 결과 획득 
  
  min_confidence = float(sizeSpin.get())          # confidence 값 변경 적용
  print(f"min_confidence : {min_confidence}")
  
  # 탐지된 객체에 대해 반복 수행 
  for i in range(0, detections.shape[2]):
    # 예측과 관련된 신뢰도 추출 
    confidence = detections[0, 0, i, 2]           # 신뢰도 추출
       
        
    if confidence > min_confidence:               # 최소 신뢰도보다 큰 경우      
      box = detections[0, 0, i, 3:7] * np.array([width, height, width, height]) # 탐지된 객체의 경계 상자 추출
      (startX, startY, endX, endY) = box.astype("int")                          # 경계 상자 좌표 추출
      print(f"confidence: {confidence}, startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")
      
      # 연관된 확률과 함께 얼굴의 경계 상자 그리기
      text = f"{confidence * 100:.2f}%"
      y = startY - 10 if startY - 10 > 10 else startY + 10  # 신뢰도 표시 위치 계산(맨 위에 표시하지 않도록 처리)
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 경계 상자 그리기
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # 신뢰도 표시
  
  # 이미지 보이기 
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 선명하게 처리 
  image = Image.fromarray(image)                  # 배열로 부터 값 변경 처리
  imgtk = ImageTk.PhotoImage(image=image)
  detection.config(image=imgtk)
  detection.image = imgtk
  
#################################################################################  

main = Tk()
main.title(title_name)
main.geometry()

img = cv2.imread(file_name)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 선명하게 처리 
image = Image.fromarray(image)                # 배열로 부터 값 변경 처리
imgtk = ImageTk.PhotoImage(image=image)
(height, width) = img.shape[:2]

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

sizeLabel=Label(main, text='Min Confidence : ')                
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)

Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))

detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)
detectAndDisplay(img, width, height)


main.mainloop()