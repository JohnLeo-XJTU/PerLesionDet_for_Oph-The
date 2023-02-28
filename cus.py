#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 06:42:30 2018
@author: pc
"""
import shutil
import os
from pyecharts.charts import Bar
import os.path
import xml.dom.minidom
import xml.etree.cElementTree as ET
from scipy.ndimage import measurements
import numpy as np

# path = "Annotations"
# files = os.listdir(path)
# s = []

remove_list = ['Paving_Stone_Degeneration','White_without_Pressure','Lattice_Degeneration','Retinal_Breaks','Retinal_Detachment']
# remove_list = ['White_without_Pressure','Degeneration','Exudatez','Retinal_Breaks']
remove_list = set(remove_list)
# =============================================================================
# extensional filename
# =============================================================================
# def file_extension(path):
#     return os.path.splitext(path)[1]
#
# for xmlFile in files:
#     if not os.path.isdir(xmlFile):
#         if file_extension(xmlFile) == '.xml':
#             print(os.path.join(path, xmlFile))
#             tree = ET.parse(os.path.join(path, xmlFile))
#             root = tree.getroot()
#             # filename = root.find('filename').text
#             #            print("filename is", path + '/' + xmlFile)
#             for Object in root.findall('object'):
#                 name=Object.find('name').text
#                 if name in remove_list:
#                     root.remove(Object)
#             tree.write('2/' + xmlFile, "utf-8")



origin_ann_dir = 'VOCdevkit/VOC2007/Annotations/'  # 设置原始标签路径为 Annos
new_ann_dir = 'VOCdevkit/VOC2007/1/'  # 设置新标签路径 Annotations
origin_pic_dir = 'VOCdevkit/VOC2007/JPEGImages/'
new_pic_dir = 'VOCdevkit/VOC2007/2/'



if not os.path.exists(new_ann_dir):
    os.makedirs(new_ann_dir)
if not os.path.exists(new_pic_dir):
    os.makedirs(new_pic_dir)



k = 0
p = 0
q = 0

for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):  # os.walk遍历目录名
    for filename in filenames:
        # k=k+1
        # print(k)
        # if os.path.isfile(r'%s%s' %(origin_ann_dir, filename)):   # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
        origin_ann_path = os.path.join(r'%s%s' % (origin_ann_dir, filename))  # 如果是，获取绝对路径（重复代码）
        new_ann_path = os.path.join(r'%s%s' % (new_ann_dir, filename))
        tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
        root = tree.getroot()  # 获取根节点
        # if len(root.findall('object')):
        ite = []
        for Object in root.findall('object'):

            name = Object.find('name').text
            ite.append(name)

            if len(set(ite) & remove_list) > 0:
            # if name in remove_list:
                print(filename)
                old_xml = origin_ann_dir + filename
                new_xml = new_ann_dir + filename
                old_pic = origin_pic_dir + filename.replace("xml","jpg")
                new_pic = new_pic_dir + filename.replace("xml","jpg")
                q = q + 1

                shutil.copy(old_xml, new_xml)
                shutil.copy(old_pic, new_pic)
                break

# print("move, ",p)
print("move, ", q)