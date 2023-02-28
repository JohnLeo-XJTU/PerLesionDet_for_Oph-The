import cv2
import numpy as np
 
import xml.dom.minidom
import os
import argparse

if not os.path.exists('./final_pic/'):
    os.makedirs('./final_pic/')

def main():
    # 存放有预测框图像的地址
    img_path = './img_out/'
    # img_path = 'fail/'
    # XML文件的地址
    anno_path = 'VOCdevkit/VOC2007/valxml/'
    # 存最终结果的文件夹
    cut_path = './final_pic/'
    # cut_path = 'failj/'
    # 获取文件夹中的文件
    imagelist = os.listdir(img_path)
 
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        img_file = img_path + image
        # 读取图片
        img = cv2.imread(img_file)
        xml_file = anno_path + image_pre + '.xml'
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xml_file)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 得到标签名为object的信息
        objects = collection.getElementsByTagName("object")
        print('done')
        for object in objects:
            # print("start")
            # 每个object中得到子标签名为name的信息
            namelist = object.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = xmin.childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = ymin.childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = xmax.childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = ymax.childNodes[0].data
            xmin = int(xmin_data)
            xmax = int(xmax_data)
            ymin = int(ymin_data)
            ymax = int(ymax_data)
            # img_cut = img[ymin:ymax, xmin:xmax, :]  # 截取框住的目标
            # cv2.imwrite(cut_path + 'cut_img_{}.jpg'.format(image_pre), img)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0, 255), thickness=2)  # 框 bgr
            cv2.putText(img, objectname, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), thickness=1)
            cv2.imwrite(cut_path + '{}.jpg'.format(image_pre), img)
 
 
if __name__ == '__main__':
    main()

