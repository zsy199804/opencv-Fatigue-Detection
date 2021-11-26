# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
# import numpy
import argparse
import imutils
import time
import dlib
import cv2
import winsound
from PIL import Image, ImageDraw, ImageFont
def cv2ImgAddText(img, text, left, top, textColor, textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar




# 眼睛长宽比
EYE_AR_THRESH = 0.2
# 眼睛闪烁阈值
EYE_AR_CONSEC_FRAMES = 3
# 打哈欠长宽比
MAR_THRESH = 0.8
# 打哈欠闪烁阈值
MOUTH_AR_CONSEC_FRAMES = 3
# 初始化
COUNTER = 0
OPEN=0
TOTAL = 0
mCOUNTER = 0
mTOTAL = 0
d=0

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')

# 获取面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #左眼
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #右眼
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"] #嘴巴

# cv2打开 本地摄像头
'''使用cap = cv2.VideoCapture(0)调用摄像头的时候出现的时候出现[ WARN:0] Failed to set mediaType (stream 0, (640x480 @ 30)
   MFVideoFormat_RGB24(unsupported media type)的错误，改变0变成700就成功了，是因为电脑摄像头配置问题，如果你出现了类似问题，更改数值。
'''
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(700)
ret = cap.isOpened()
fps = cap.get(5)/10000

# 视频流循环帧
while ret:
    # 进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    tstep = cap.get(1)
    iloop = fps / 2  # 每秒处理2帧
    while iloop:
        cap.grab()  # 只取帧不解码，
        iloop = iloop - 1
        # if iloop < 1:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)

    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)

        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)

        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 构造函数计算左右眼的EAR值
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # 获得打哈欠的值
        mar = mouth_aspect_ratio(mouth)

        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
        # 进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 5)


        # 循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.2
            COUNTER += 1
            d+=1
            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER > EYE_AR_CONSEC_FRAMES:  # 阈值：3
                TOTAL += 1
                COUNTER = 0

        elif ear > EYE_AR_THRESH and d>=3:
            OPEN+=1

            # 如果连续3次都大于阈值，则表示进行了一次睁眼
            if OPEN>=5:  # 阈值：4
                TOTAL-=1
                if TOTAL<=0:
                    COUNTER = 0
                    d=0
                    TOTAL = 0
                OPEN = 0
        # 同理，判断是否打哈欠
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
            # cv2.putText(frame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
                mCOUNTER = 0



        # 进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # cv2.putText将眨眼次数进行显示
        frame=cv2ImgAddText(frame, "闭眼: {}".format(TOTAL), 50, 10 , (255, 0, 0), 25)
        frame=cv2ImgAddText(frame, "闭眼时间: {}".format(COUNTER), 200, 10,  (255, 0, 0), 25)
        frame=cv2ImgAddText(frame,"睁眼:{}".format(OPEN), 350,10,(255,0,0),25)
        frame=cv2ImgAddText(frame, "实时眼睛大小: {:.2f}".format(ear), 500, 10,  (255, 0, 0), 25)
        # cv2.putText将打哈欠次数进行显示
        frame = cv2ImgAddText(frame, "打哈欠: {}".format(mTOTAL), 50, 35, (255, 0, 0),25)
        frame = cv2ImgAddText(frame, "张嘴时间: {}".format(mCOUNTER), 200, 35,(255, 0, 0), 25)
        frame = cv2ImgAddText(frame, "实时嘴巴大小: {:.2f}".format(mar), 350, 35, (255, 0, 0), 25)

        # print('眼睛实时长宽比:{:.2f} '.format(ear))
        # print('嘴巴实时长宽比:{:.2f} '.format(mar))

    if TOTAL >= 3:
        cv2.putText(frame, "SLEEP", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # 控制电脑蜂鸣器
        winsound.Beep(2222,111)
    else:
        cv2.putText(frame, "WAKE", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if mTOTAL>=1:
        cv2.putText(frame, "WARN!", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # cv2.putText(frame, "Press 'x': Quit", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 159), 2)
        frame=cv2ImgAddText(frame, "按x退出警告", 200, 300, (255, 0, 0), 25)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            mTOTAL=0


    # cv2.putText(frame, "按q退出程序", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 159), 2)
    frame = cv2ImgAddText(frame, "按q退出程序", 20, 500, (255, 255, 0), 40)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()

