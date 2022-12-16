import os
import shutil

path = '/home/liaogl/ALL/'
new_path = '/home/liaogl/1-21/'

for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        # print(files[i])
        if (files[i][-3:] == 'jpg') or (files[i][-3:] == 'png') or (files[i][-3:] == 'JPG'):
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + files[i]
            shutil.copy(file_path, new_file_path)

        # yn_close = input('是否退出？')
