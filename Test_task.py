import torch
import cv2 as cv
import pandas as pd
import numpy as np
import time


def handler(frame, class_name, model):
    detected_frame = model(frame)
    df = pd.DataFrame(detected_frame.pandas().xyxy[0])

    # Удаление записей с не заданным пользователем именем класса
    indexes = df[df['name'] != class_name].index
    df.drop(indexes, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Если нужного объекта в кадре нет, возвращается обычный кадр
    if (df.empty):
        return frame

    # Координаты прямоугольной рамки
    x1, y1 = int(df['xmin'][0]), int(df['ymin'][0])
    x2, y2 = int(df['xmax'][0]), int(df['ymax'][0])

    # Маска для прямоугольника в кадре
    mask = np.zeros_like(frame)
    cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    frame_masked = cv.bitwise_and(frame, mask)

    return frame_masked


def video_processing(video_path, output_name, class_name, model):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Видеофайл не открылся!")
        print("Повторите ВВОД!")
        return True

    print("Видеофайл открылся, выполняется обработка...")

    # Параметры для записи видеофайла
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter(output_name,
                         fourcc,
                         fps, (width, height),
                         isColor=True)

    # Покадровая обработка и запись
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = handler(frame, class_name, model)
        out.write(frame)

    cap.release()
    cv.destroyAllWindows()
    return False


def main():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    output_name = "ResultVideo.mp4"
    class_name = input("Введите имя класса объекта: ")

    flag = True
    while (flag):
        video_path = input("Введите путь к видеофайлу: ")

        begin_time = time.time()
        flag = video_processing(video_path, output_name, class_name, model)
        end_time = time.time()

    print(f"Обработка видео заняла: {end_time - begin_time} сек.")
    print(f"Новый видеофайл сохранён в директорию программы с именем {output_name}")


if __name__ == "__main__":
    main()