import cv2
import numpy as np

from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# 파일명일 때 이름 조심해서 사용( \r 같은 문자가 있으면 안됨)
model_name = "D:/Github/Vision_WS/OpenCV_Part1/data/dnn/res10_300x300_ssd_iter_140000.caffemodel"  # 모델 파일
prototxt_name = "D:\Github\Vision_WS\OpenCV_Part1\data\dnn\deploy.prototxt.txt"  # prototxt 파일
min_confidence = 0.5  # 최소 신뢰도(임계값 조정)
file_name = "D:/Github/Vision_WS/OpenCV_Part1/videos/obama_01.mp4"  # 파일 이름

title_name = 'dnn Deep Learnig object detection Video'
frame_width = 300
frame_height = 300
cap = cv2.VideoCapture()

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "videos", title = "Select file",filetypes = (("MP4 files","*.mp4"),("all files","*.*")))
    print('File name : ', file_name)
    global cap                          # 전역 변수 사용
    cap = cv2.VideoCapture(file_name)   # 새로운 파일로 비디오 캡쳐 시작
    detectAndDisplay()                  # 새로운 파일로 비디오 캡쳐 시작

def detectAndDisplay():
  _, frame = cap.read()                # 비디오 스트림에서 프레임 읽기
  (h, w) = frame.shape[:2]             # 프레임의 높이와 너비 추출
  # blob을 모델에 전달하고 탐지 결과를 획득
  model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)  # 모델 불러오기

  # 이미지 크기 변경하고 정규화 수행
  blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
  )
  model.setInput(blob)          # blob을 모델에 전달
  detections = model.forward()  # 탐지 결과 획득

  min_confidence = float(sizeSpin.get())          # confidence 값 변경 적용

  # 탐지된 객체에 대해 반복 수행
  for i in range(0, detections.shape[2]):
    # 예측과 관련된 신뢰도 추출
    confidence = detections[0, 0, i, 2]  # 신뢰도 추출

    if confidence > min_confidence:  # 최소 신뢰도보다 큰 경우만 처리 
      (height, width) = frame.shape[:2]      
      # 탐지된 객체의 경계 상자 추출
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  
      (startX, startY, endX, endY) = box.astype("int")  # 경계 상자 좌표 추출
      print(f"confidence: {confidence}, startX: {startX}, startY: {startY}, endX: {endX}, endY: {endY}")

      # 연관된 확률과 함께 얼굴의 경계 상자 그리기
      text = f"{confidence * 100:.2f}%"
      y = startY - 10 if startY - 10 > 10 else startY + 10  # 신뢰도 표시 위치 계산(맨 위에 표시하지 않도록 처리)
      cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
      cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      
  # 동영상 보여주기
  cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)    # OpenCV image를 PIL image로 변환
  img = Image.fromarray(cv2image)                       # PIL image를 ImageTk 객체로 변환
  imgtk = ImageTk.PhotoImage(image=img)                 # ImageTk 객체를 imgtk로 변환
  lmain.imgtk = imgtk                                   # 레이블에 새로운 이미지 지정
  lmain.configure(image=imgtk)                          # 레이블에 새로운 이미지 지정
  lmain.after(10, detectAndDisplay)                     # 10ms 후에 detectAndDisplay 함수 호출

#main
main = Tk()
main.title(title_name)
main.geometry()

#Graphics window
label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

sizeLabel=Label(main, text='Min Confidence : ')
sizeLabel.grid(row=1,column=0)
sizeVal  = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)

Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))

imageFrame = Frame(main)
imageFrame.grid(row=2,column=0,columnspan=4)
  
# 캡쳐된 비디오 프레임 표시
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()  # main loop
