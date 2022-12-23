import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
import cv2
import os
import main

# 创建一个界面窗口
win = tkinter.Tk()
win.title("车牌识别")
win.geometry("800x600")

# 设置全局变量
original = Image.new('RGB', (300, 400))
var = tkinter.StringVar()
select_file =' '


# 实现在本地电脑选择图片
def choose_file():
    var.set('')
    global select_file
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    load = Image.open(select_file)
    load = load.resize((250, 250), Image.LANCZOS)
    # 声明全局变量
    global original
    original = load
    # 展示图片
    render = ImageTk.PhotoImage(load)
    img = tkinter.Label(win, image=render)
    img.image = render
    img.place(x=100, y=100)


# 实现车牌识别输出功能
def hit_me():
    if select_file == ' ':
        tkinter.messagebox.showwarning('Error', '请先选择图片!')
    else:

        origin_image = cv2.imread(select_file)
        image = origin_image.copy()
        carLicense_image = main.get_carLicense_img(image, origin_image)
        image = carLicense_image.copy()
        word_images = main.carLicense_spilte(image)
        word_images_ = word_images.copy()
        result = main.template_matching(word_images_)
        var.set(result)


# 设置选择图片,识别车牌,退出的按钮
button0 = tkinter.Button(win, text="    退出    ", command=win.quit)
button0.pack(side='bottom')

button1 = tkinter.Button(win, text="选择图片", command=choose_file)
button1.pack(side='bottom')

button2 = tkinter.Button(win, text="识别车牌", command=hit_me)
button2.pack(side='bottom')
# 设置标签分别为图片和车牌号，以及输出车牌号的标签
label1 = tkinter.Label(win, text="图片")
label1.place(x=200, y=50)

label2 = tkinter.Label(win, text="车牌号")
label2.place(x=600, y=50)

label3 = tkinter.Label(win, textvariable=var, font=('Arial', 12))
label3.place(x=550, y=250)

win.mainloop()
