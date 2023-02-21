import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import random

cwd = os.getcwd()
data_path = '/data/CUB_200_2011'
savedir = './'
dataset_list = ['base','val','novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(join(data_path, 'train'))]

classfile_list_all = []
for i, folder in enumerate(folder_list):
    classfiles = []
    for split in ['train', 'test']:
        folder_path = join(join(data_path, split), folder)
        classfiles += [join(split, join(folder, cf)) for cf in listdir(folder_path)]
    classfile_list_all.append(classfiles)
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)