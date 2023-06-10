import argparse
import configparser
from tkinter import *
import cv2

from PIL import Image
from PIL import ImageTk
from random import randrange

root = Tk()


def add_line(event):
    global click_n
    global lines
    global x1, y1
    if click_n % 2 == 0:
        x1 = event.x
        y1 = event.y
    else:
        x2 = event.x
        y2 = event.y
        # Draw the line in the given co-ordinates
        lines.append((x1, y1, x2, y2))
        canvas.create_line(x1, y1, x2, y2, fill="green", width=5)
    click_n += 1


def save_config(event):
    global lines
    global conf_name
    global height_coef
    global width_coef
    config_object = configparser.ConfigParser()
    with open(conf_name, 'w') as f:
        for i in lines:
            points = []
            x_1 = int(i[0] * width_coef)
            x_2 = int(i[2] * width_coef)
            y_1 = int(i[1] * height_coef)
            y_2 = int(i[3] * height_coef)
            points.append({"points": ((x_1, y_1), (x_2, y_2)),
                           "color": (randrange(0, 255), randrange(0, 255), randrange(0, 255)),
                           "thickness": 2
                           })
        config_object['Lines'] = {"values": points}
        config_object.write(f)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True, help='path to the video')
    ap.add_argument('-n', '--name', required=True, help='path of the config file to save')
    args = vars(ap.parse_args())
    lines = []
    conf_name = args['name']
    vs = cv2.VideoCapture(args['video'])
    print(type(vs.read()[1]), vs.read()[1].shape)
    frame = Image.fromarray(vs.read()[1])
    width = frame.size[0]
    height = frame.size[1]
    print(width, height)
    new_width = 1000
    new_height = int(new_width * height / width)
    height_coef = height / new_height
    width_coef = width / new_width
    frame = frame.resize((new_width, new_height), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(frame)
    canvas = Canvas(root, width=new_width, height=new_height)
    canvas.create_image(0, 0, image=image, anchor=NW)
    b1 = Button(text="save",
                width=15, height=3)
    b1.pack()
    canvas.bind("<Button-1>", add_line)
    b1.bind('<Button-1>', save_config)
    canvas.pack()
    click_n = 0

    root.mainloop()
