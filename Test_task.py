import torch
import cv2 as cv
import pandas as pd
import numpy as np
import os

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def one_img(): 
    img = "people.jpg"

    results = model(img)
    results.print()
    print("\n\n\n")

    df = pd.DataFrame(results.pandas().xyxy[0], )
    print(df)

    indexes = df[df['class'] != 0].index
    df.drop(indexes, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)

    # for i in range(df.shape[0]):
    #     print(df['xmin'][i], ':', df['ymin'][i])
    #     print(df['xmax'][i], ':', df['ymax'][i])

    x1,y1 = int(df['xmin'][0]), int(df['ymin'][0])
    x2,y2 = int(df['xmax'][0]), int(df['ymax'][0])
    image = cv.imread(img)
    mask = np.zeros_like(image)
    cv.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)

    img_masked = cv.bitwise_and(image, mask)

    cv.imshow('masked', img_masked)
    cv.waitKey(0)

def handler(frame):
    detected_frame = model(frame)
    df = pd.DataFrame(detected_frame.pandas().xyxy[0])

    indexes = df[df['class'] != 0].index
    df.drop(indexes, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if(df.empty):
        return frame
    
    x1,y1 = int(df['xmin'][0]), int(df['ymin'][0])
    x2,y2 = int(df['xmax'][0]), int(df['ymax'][0])
    image = frame

    mask = np.zeros_like(image)
    cv.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)

    img_masked = cv.bitwise_and(image, mask)

    return img_masked

    

def video_capture():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = handler(frame)
        cv.imshow('video', frame)
        

        key = cv.waitKey(1)
        if  key == 27:
            break
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    #one_img()
    video_capture()