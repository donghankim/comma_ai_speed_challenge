import cv2
import numpy as np
import os, pdb, csv, re

def get_frames(vid_path, target):
    cap = cv2.VideoCapture(vid_path)
    success,image = cap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(target, f"frame_{count}.jpg"), image)
        success,image = cap.read()
        print(f"Reading frame{count}: ", success)
        count += 1

    cap.release()

def read_speed(speed_path):
    speeds = []
    with open(speed_path) as file_obj:
        for line in file_obj:
            speeds.append(float(line))

    return speeds

def create_csv(image_path, speed_arr, csv_path):
    with open(csv_path, 'w', newline = '') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["frame_name", "speed"])
        image_names = os.listdir(image_path)
        image_names.sort(key=lambda f: int(re.sub('\D', '', f)))

        for i in range(len(image_names)):
            try:
                writer.writerow([image_names[i], speed_arr[i]])
            except Exception as e:
                # just in case index out of range
                print(e, i)

    file_obj.close()





