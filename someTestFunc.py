from PIL import Image
import os
import cv2

def resize_image():
    image_path = 'D:/alex/CascadeNetwork3/testing/images/2039717921.jpg'
    img = Image.open(image_path)
    img = img.resize((960, 544))
    img.save('D:/alex/CascadeNetwork3/testing/images/2039717921_resized.jpg')


def drawTrueLandMark():
    testimages = 'D:/alex/CascadeNetwork3/'
    txt = 'D:/alex/CascadeNetwork3/images1/gt.txt'
    lines = []
    with open(txt) as f:
        lines = f.readlines()
    index = 0
    for line in lines:
        ln = line.split(' ')
        if ln[0][8] != '2':
            continue
        index = index + 1
        img_path = testimages + ln[0]
        img = cv2.imread(img_path)
        x1 = float(ln[1])
        y1 = float(ln[2])
        x2 = float(ln[3]) + x1
        y2 = float(ln[4]) + y1
        x11 = float(ln[5])
        y11 = float(ln[6])
        x22 = float(ln[7])
        y22 = float(ln[8])
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,0,255))
        cv2.circle(img, (int(x11), int(int(y11))), 3, (0, 0, 255))
        cv2.circle(img, (int(x22), int(int(y22))), 3, (0, 0, 255))
        savePath = os.path.join(testimages, 'testing', 'ori_imgs')
        cv2.imwrite(os.path.join(savePath, "result_%d.jpg" % (index)), img)
        print(index)


def splitVideo():
    videoPath = "D:/alex/CascadeNetwork3/875bbd153bc5844fc203f7e4b38de6cd.mp4"
    savePath = "D:/alex/CascadeNetwork3/testImages/"
    cap = cv2.VideoCapture(videoPath)
    index = 0
    while True:
        state, frame = cap.read()
        if not state:
            break
        cv2.imwrite(savePath + str(index) + '.jpg', frame)
        index = index + 1


splitVideo()