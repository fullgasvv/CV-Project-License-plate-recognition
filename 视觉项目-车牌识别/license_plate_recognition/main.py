import cv2
import os
import numpy as np


'''def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# plt显示彩色图片
def plt_show0(img):#opencv以bgr格式读入图片，改为rgb格式输出图片
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()'''


# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0) #高斯滤波去噪
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#灰度化函数
    return gray_image


def get_carLicense_img(image, origin_image):
    gray_image = gray_guss(image)
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0) #sobel算子边缘检测
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)#二值化图像
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 3)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 2))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18))
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.medianBlur(image, 15)#以上为膨胀和腐蚀操作
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#车牌是矩形形状，在膨胀腐蚀后的图画最小正接矩形
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 3)) and (weight < (height * 4)):
            image = origin_image[y:y + height, x:x + weight]
            return image
#因为我国车牌长宽比一般为3.14，所以做一个判断就可以找出包围车牌的正确矩形


#车牌字符分割
def carLicense_spilte(image):
    gray_image = gray_guss(image)
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    #plt_show(image)
    area_white = 0
    area_black = 0
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    if area_white > area_black:
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    words = sorted(words, key=lambda s: s[0], reverse=False)
    i = 0
    for word in words:
        if (word[3] > (word[2] * 1.8)) and (word[3] < (word[2] * 3.5)):
            i = i + 1
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            word_images.append(splite_image)
    return word_images


#image = carLicense_image.copy()
#word_images = carLicense_spilte(image)



#以下是模板匹配步骤

template = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',
            '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']

# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

# 中文模板列表
def get_chinese_words_list():
    chinese_words_list = []
    for i in range(34,64):
        c_word = read_directory('./refer/'+ template[i])
        chinese_words_list.append(c_word)
    return chinese_words_list
chinese_words_list = get_chinese_words_list()

# 英文模板列表
def get_eng_words_list():
    eng_words_list = []
    for i in range(10,34):
        e_word = read_directory('./refer/'+ template[i])
        eng_words_list.append(e_word)
    return eng_words_list
eng_words_list = get_eng_words_list()


# 英文数字模板列表
def get_eng_num_words_list():
    eng_num_words_list = []
    for i in range(0,34):
        word = read_directory('./refer/'+ template[i])
        eng_num_words_list.append(word)
    return eng_num_words_list
eng_num_words_list = get_eng_num_words_list()


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    template_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    image_ = image.copy()
    height, width = image_.shape
    template_img = cv2.resize(template_img, (width, height))
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]


#由前至后读取车牌字符并与标准模板进行匹配，取得分最高者为结果
def template_matching(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        if index==0:
            best_score = []
            for chinese_words in chinese_words_list:
                score = []
                for chinese_word in chinese_words:
                    result = template_score(chinese_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[34+i])
            r = template[34+i]
            results.append(r)
            continue
        if index==1:
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                for eng_word in eng_word_list:
                    result = template_score(eng_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[10+i])
            r = template[10+i]
            results.append(r)
            continue
        else:
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word,word_image)
                    score.append(result)
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[i])
            r = template[i]
            results.append(r)
            continue
    return results


#word_images_ = word_images.copy()
#result = template_matching(word_images_)
#print(result)


