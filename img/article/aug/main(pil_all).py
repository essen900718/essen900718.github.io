import cv2
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def yolobbox2bbox(x, y, w, h, dw, dh):
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    return (l, t, r, b)

def bbox2yolobbox(box, dw, dh):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x / dw
    w = w / dw
    y = y / dh
    h = h / dh
    return (round(x, 6), round(y, 6), round(w, 6), round(h, 6))

def read_file(file, dw, dh):
    dic = {}
    for line in file.readlines():
        box = yolobbox2bbox(float(line.split(" ")[1]), float(line.split(" ")[2]), float(line.split(" ")[3]), float(line.split(" ")[4]), dw, dh)
        dic[box] = int(line.split(" ")[0])
    return dic

def drawredpoint(img, box):
    draw = ImageDraw.Draw(img)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], fill="#ff0000")
    return img

def getbboxfromimage(image):
    red = [0,0,255]
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    Y2, X2 = np.where(np.all(img==red,axis=2))
    x1 = min(X2)
    y1 = min(Y2)
    x2 = max(X2)
    y2 = max(Y2)

    return x1, y1, x2, y2

def writeback(file_name, dic, image, image_path, label_path):
    new_image_path = image_path + file_name + '.jpg'
    new_label_path = label_path + file_name + '.txt'

    f1 = open(new_label_path, 'w')

    for box in dic:
        line = str(dic[box]) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + '\n'
        print(line)
        f1.write(line)
    f1.close()
    image.save(new_image_path)    

def aug_matrix(skew_type, w, h):

    x1 = 0
    x2 = h
    y1 = 0
    y2 = w

    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
    
    max_skew_amount = max(w, h)
    max_skew_amount = int(math.ceil(max_skew_amount * magnitude))
    skew_amount = random.randint(1, max_skew_amount)

    skew = skew_type

    if skew_type == "RANDOM":
        skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
    else:
        skew = skew_type

    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":
        if skew == "TILT":
            skew_direction = random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = random.randint(2, 3)
        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left
    if skew == "CORNER":
        skew_direction = random.randint(0, 7)
        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

    if skew_type == "ALL":
        # Not currently in use, as it makes little sense to skew by the same amount
        # in every direction if we have set magnitude manually.
        # It may make sense to keep this, if we ensure the skew_amount below is randomised
        # and cannot be manually set by the user.
        corners = dict()
        corners["top_left"] = (y1 - random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["top_right"] = (y2 + random.randint(1, skew_amount), x1 - random.randint(1, skew_amount))
        corners["bottom_right"] = (y2 + random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
        corners["bottom_left"] = (y1 - random.randint(1, skew_amount), x2 + random.randint(1, skew_amount))
        new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

    matrix = []

    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    return perspective_skew_coefficients_matrix

def do(image, matrix):
    return image.transform(image.size, Image.PERSPECTIVE, matrix, resample=Image.BICUBIC)

# 調整擴增參數
skew_type = 'TILT_TOP_BOTTOM' 
# "ALL", "RANDOM", "TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER", "TILT_TOP_BOTTOM"
magnitude = 0.3

label_path = 'C:/Users/user/Desktop/aug/label_org/'
image_path = r'C:/Users/user/Desktop/aug/image_org/'
newimagepath = r'C:/Users/user/Desktop/aug/image_tilt/'
newlabelpath = 'C:/Users/user/Desktop/aug/label_tilt/'

files = os.listdir(image_path)

for file in files:

    file_name = file.split('.')[0]
    print(file_name)

    newfilename = file_name + '_aug'

    img = cv2.imread(image_path + file_name + '.jpg')
    rows, cols, ch = img.shape
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    f1 = open(label_path + file_name + '.txt', 'r')
    dic = read_file(f1, cols, rows)
    f1.close()

    new_dic = {}
    matrix = aug_matrix(skew_type, cols, rows)

    for box in dic:
        print(box)
        image_org = img.copy()
        image_org = drawredpoint(image_org, box)
        image_tilt = do(image_org, matrix)
        x1_tilt, y1_tilt, x2_tilt, y2_tilt = getbboxfromimage(image_tilt)
        yolobbox_tilt = bbox2yolobbox((x1_tilt, y1_tilt, x2_tilt, y2_tilt), cols, rows)
        new_dic[yolobbox_tilt] = dic[box]

    newimage = img.copy()
    newimage = do(newimage, matrix)
    writeback(newfilename, new_dic, newimage, newimagepath, newlabelpath)
