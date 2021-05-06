import numpy as np
import cv2

import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
lang = 'jpn'
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def is_inside(contour, point):
    x = point[0]
    y = point[1]
    xp= point[0]+point[2]
    yp= point[1]+point[3]
    if cv2.pointPolygonTest(contour, (x,y), False) < 0:
        return False
    if cv2.pointPolygonTest(contour, (x,yp), False) < 0:
        return False
    if cv2.pointPolygonTest(contour, (xp,y), False) < 0:
        return False
    if cv2.pointPolygonTest(contour, (xp,yp), False) < 0:
        return False
    return True

def find_text_dialog(img):
    imgcp = img.copy()
    imgh, imgw, _ = imgcp.shape
    print('width:  ', imgw)
    print('height: ', imgh)
    rectlist = []
    approxlist = []
    imgGrey = cv2.cvtColor(imgcp, cv2.COLOR_BGR2GRAY)
    _, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        approxfine = cv2.approxPolyDP(contour, 0.001* cv2.arcLength(contour, True), True)
        #cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) >= 4 and len(approx) <= 8 and cv2.contourArea(contour) > (imgw*0.05)*(imgh*0.05) and cv2.contourArea(contour) < (imgw*0.5)*(imgh*0.5):
            xe,ye,we,he = cv2.boundingRect(approx)
            cv2.rectangle(imgcp, (xe,ye), (xe+we+5, ye+he+5), (255, 0, 0), 1)
            rectlist.append([xe,ye,we,he])
            approxlist.append(approxfine)
            cv2.drawContours(imgcp, [approxfine], 0, (0, 0, 255), 2)
            cv2.putText(imgcp, "Dialog(R)", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        if len(approx) > 8 and len(approx) < 40 and cv2.contourArea(contour) > 50*50 and cv2.contourArea(contour) < 200*200:
            xe,ye,we,he = cv2.boundingRect(approx)
            cv2.rectangle(imgcp, (xe,ye), (xe+we+5, ye+he+5), (255, 0, 0), 1)
            rectlist.append([xe,ye,we,he])
            approxlist.append(approxfine)
            cv2.drawContours(imgcp, [approxfine], 0, (0, 0, 255), 2)
            cv2.putText(imgcp, "Dialog(C)", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    return approxlist, rectlist, imgcp

def find_text_test_case(img):
    imgcp = img.copy()
    boxes = pytesseract.image_to_data(imgcp, lang = lang)
    for a,b in enumerate(boxes.splitlines()):
         if a!=0:
             b = b.split()
             if len(b)==12:
                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                 #cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
                 cv2.rectangle(imgcp, (x-5,y-5), (x+w+5, y+h+5), (255, 255, 255), -1)
    return imgcp

def find_text(img, contourrect):
    imgcp = img.copy()
    imgcp2 = img.copy()
    textlist = []
    i = 0
    for rect in contourrect:
        i += 1
        im = imgcp[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv2.rectangle(imgcp2, (rect[0],rect[1]), (rect[0]+rect[2]+5, rect[1]+rect[3]+5), (255, 0, 0), 1)
        boxes = pytesseract.image_to_data(im, lang = lang)
        for a,b in enumerate(boxes.splitlines()):
             if a!=0:
                 b = b.split()
                 if len(b)==12:
                     x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                     textlist.append([x+rect[0],y+rect[1],w,h])
                     #cv2.imshow("im" + str(i), im)
                     #cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
                     cv2.rectangle(imgcp2, (x+rect[0],y+rect[1]), (x+rect[0]+w+5, y+rect[1]+h+5), (255, 0, 255), 1)
    return textlist, imgcp2

def clean_text(img, contourlist, textarealist):
    cleanimg = img.copy()
    for contour in contourlist:
        for textarea in textarealist:
            if is_inside(contour, textarea):
                cv2.drawContours(cleanimg, [scale_contour(contour, 0.90)], 0, (255, 255, 255), -1)
                cv2.rectangle(cleanimg, (textarea[0],textarea[1]), (textarea[0]+textarea[2]+5, textarea[1]+textarea[3]+5), (255, 255, 255), -1)
    return cleanimg

def main():
    print("run")
    img = cv2.imread('.jpg') #INPUT PICTURE FILE
    textdialog, textdialogrect, img1 = find_text_dialog(img)
    textarea, img2 = find_text(img, textdialogrect)
    test_case = find_text_test_case(img)
    output = clean_text(img, textdialog, textarea)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("test case", test_case)
    cv2.imshow("output", output)
    if not cv2.imwrite("outputImage.png", output):
        print("save not successful")
    else:
        print("save successfully")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
