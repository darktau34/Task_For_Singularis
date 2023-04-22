import torch
import cv2 as cv
import pandas as pd
import numpy as np
import time

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def one_img():
    img = "screen1.png"

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

    x1, y1 = int(df['xmin'][0]), int(df['ymin'][0])
    x2, y2 = int(df['xmax'][0]), int(df['ymax'][0])
    image = cv.imread(img)
    mask = np.zeros_like(image)
    cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    img_masked = cv.bitwise_and(image, mask)

    cv.imshow('masked', img_masked)
    cv.waitKey(0)


def handler(frame):
    detected_frame = model(frame)
    df = pd.DataFrame(detected_frame.pandas().xyxy[0])

    indexes = df[df['class'] != 2].index
    df.drop(indexes, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if (df.empty):
        return frame

    x1, y1 = int(df['xmin'][0]), int(df['ymin'][0])
    x2, y2 = int(df['xmax'][0]), int(df['ymax'][0])
    image = frame

    mask = np.zeros_like(image)
    cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

    img_masked = cv.bitwise_and(image, mask)

    return img_masked


def video_capture():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = handler(frame)
        cv.imshow('video', frame)

        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def video_processing():
    video_path = "drift.mp4"
    output_name = "MyVideo.mp4"
    cap = cv.VideoCapture(video_path)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(output_name,
                         fourcc,
                         fps, (width, height),
                         isColor=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = handler(frame)
        out.write(frame)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # one_img()
    # video_capture()
    begin_time = time.time()
    video_processing()
    end_time = time.time()
    print(f"Video processing time: {end_time - begin_time}")