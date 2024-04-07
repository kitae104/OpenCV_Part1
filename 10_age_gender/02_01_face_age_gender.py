import numpy as np
import cv2
import time
from tkinter import *                       # GUI를 위한 라이브러리
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

face_model = 'D:\Github\Vision_WS\OpenCV_Part1/model/dnn/res10_300x300_ssd_iter_140000.caffemodel'                    # 얼굴 검출 모델
face_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1/model/dnn/deploy.prototxt.txt'                                      # 얼굴 검출 모델(메타 정보)
age_model = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/age_net.caffemodel'             # 나이 검출 모델
age_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/deploy_age.prototxt'         # 나이 검출 모델(메타 정보)
gender_model = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/gender_net.caffemodel'       # 성별 검출 모델
gender_prototxt = 'D:\Github\Vision_WS\OpenCV_Part1\model\dnn/deploy_gender.prototxt'   # 성별 검출 모델(메타 정보)
image_file = 'D:/Github/Vision_WS/OpenCV_Part1/images/son-align/1.jpg'                  # 이미지 파일

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] # 나이 리스트 
gender_list = ['Male','Female'] # 성별 리스트

title_name = 'Age and Gender Recognition'   # 윈도우 타이틀 이름
min_confidence = 0.5                        # 최소 확률 값  
min_likeness = 0.5                          # 최소 유사도 값
frame_count = 0                             # 프레임 카운트
recognition_count = 0                       # 인식 카운트
elapsed_time = 0                            # 경과 시간
OUTPUT_SIZE = (300, 300)                    # 출력 크기

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)              # 얼굴 검출 모델 로딩
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)            # 나이 검출 모델 로딩
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)   # 성별 검출 모델 로딩

# 파일 선택 함수
def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)      # 이미지 파일 읽기
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) # BGR을 RGB로 변환
    image = Image.fromarray(image)          # 이미지 객체 생성
    imgtk = ImageTk.PhotoImage(image=image)  # 이미지 객체를 표시하기 위한 객체 생성
    (height, width) = read_image.shape[:2]  # 이미지 크기
    fileLabel['text'] = file_name           # 파일 이름 표시
    detectAndDisplay(read_image)            # 얼굴 검출 및 나이, 성별 인식


# 얼굴 인식 함수
def detectAndDisplay(read_image):
    pass

# main 부분 
main = Tk()                         # Tk 객체 인스턴스 생성
main.title(title_name)              # 윈도우 타이틀 이름 설정
main.geometry()                     # 윈도우 크기 설정

# 입력 이미지를 로드하고 BGR에서 RGB로 변환
read_image = cv2.imread(image_file)         # 이미지 파일 읽기
(height, width) = read_image.shape[:2]      # 이미지 크기
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) # BGR을 RGB로 변환
image = Image.fromarray(image)              # 이미지 객체 생성
imgtk = ImageTk.PhotoImage(image=image)     # 이미지 객체를 표시하기 위한 객체 생성

label = Label(main, text=title_name)        # 라벨 생성
label.config(font=("Courier", 18))          # 라벨 설정
label.grid(row=0, column=0, columnspan=4)   # 라벨 위치

fileLabel = Label(main, text=image_file)        # 파일 라벨 생성(파일 이름 표시)
fileLabel.grid(row=1, column=0, columnspan=2)   # 파일 라벨 위치

# 파일 선택 버튼 생성
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(N, S, W, E))

# 사진 보이기 
detection=Label(main, image=imgtk)              # 이미지 라벨 생성
detection.grid(row=2,column=0,columnspan=4)     # 이미지 라벨 위치

# 스크롤 텍스트 생성
log_ScrolledText = tkst.ScrolledText(main, height=20) # 스크롤 텍스트 생성
log_ScrolledText.grid(row=3,column=0,columnspan=4, sticky=(N, S, W, E)) # 스크롤 텍스트 위치
log_ScrolledText.configure(font='TkFixedFont') # 스크롤 텍스트 설정

log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14)) # 스크롤 텍스트 태그 설정
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center') # 스크롤 텍스트 태그 설정

detectAndDisplay(read_image)

main.mainloop()                     # GUI 시작