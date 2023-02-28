import os.path
import glob
import xml.etree.ElementTree as ET
path = 'VOCdevkit/VOC2007/valxml/'
for xml_file in glob.glob(path + '*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        objectname = member.find('name').text
        if objectname == 'White_without_pressure':
            print(objectname)
            member.find('name').text = str('White_without_Pressure')
            tree.write(xml_file)


