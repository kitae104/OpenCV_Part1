import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# 경로를 사용할 때 절대 경로로 지정해야 함. 상대 경로로 지정하면 오류 발생할 수 있음.
face_cascade_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'D:/Github/Vision_WS/OpenCV_Part1/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
file_name = 'D:/Github/Vision_WS/OpenCV_Part1/videos/tedy_01.mp4'

title_name = 'Haar cascade 객체 감지(Video)'
frame_width = 500             # 화면에 보여줄 프레임의 너비
cap = cv2.VideoCapture()      # 비디오 캡쳐 객체 생성

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# cascade 파일 로딩
if not face_cascade.load("D:\Github\Vision_WS\OpenCV_Part1\data\haarcascades\haarcascade_frontalface_alt.xml"):
  print('--(!) face cascade를 로딩하는 과정에서 문제가 발생하였습니다.')
  exit(0)
  
if not eyes_cascade.load("D:\Github\Vision_WS\OpenCV_Part1\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml"):
  print('--(!) eyes cascade를 로딩하는 과정에서 문제가 발생하였습니다.')
  exit(0)

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./video",title = "Select file",filetypes = (("MP4 files","*.mp4"),("all files","*.*")))
    print('File name : ', file_name)
    global cap                              # 전역 변수 사용 
    cap = cv2.VideoCapture(file_name)       # 파일에서 비디오 캡쳐 
    detectAndDisplay()                      # 함수 수행 

def detectAndDisplay():
  _, frame = cap.read()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 회색으로 변환
  frame_gray = cv2.equalizeHist(frame_gray)               # 히스토그램 평활화 
  
  # 얼굴 검출
  faces = face_cascade.detectMultiScale(frame_gray)
  for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)                      # 얼굴 중심 좌표
    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # 얼굴에 사각형 그리기
    faceROI = frame_gray[y:y+h, x:x+w]                # 얼굴 영역 추출
    
    # 각 얼굴에서 눈 검출
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2,y2,w2,h2) in eyes:
      eye_center = (x + x2 + w2//2, y + y2 + h2//2) # 눈 중심 좌표
      radius = int(round((w2 + h2)*0.25))           # 눈 반지름
      frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4) # 눈에 원 그리기
      
  # 이미지 보이기 
  cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(cv2image)
  imgtk = ImageTk.PhotoImage(image=img)
  lmain.imgtk = imgtk
  lmain.configure(image=imgtk)
  lmain.after(10, detectAndDisplay)
  
# Main
main = Tk()                   # 메인 윈도우 생성
main.title(title_name)        # 타이틀 이름 지정
main.geometry()               # 크기 지정

# Window
label = Label(main, text=title_name)  # 라벨 생성
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

Button(main, text="파일 찾기", height=2, command=lambda: selectFile()).grid(row=1, column=0, columnspan=4, sticky=(W, E))
imageFrame = Frame(main)
imageFrame.grid(row=2,column=0,columnspan=4)

# video frame 
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()               # 메인 루프 실행
