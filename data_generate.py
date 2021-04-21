'''
# @Descripttion: 
# @Date: 2021-04-21 16:37:39
# @Author: cfireworks
# @LastEditTime: 2021-04-21 21:36:47
'''
import os
import cv2
import random

select_cls = ["blq0", "dyhgq0", "dz0", "jddz0", "kg0", "dk0", "sjbyq0", "ljbyq0", "dr0", "fdj", "jdb"]

def select_imgs(img_dir, out_dir, select_cls, count_per_cls=200):
    '''
    # @description: 
    # @param {*}
    # @return {*}
    '''
    sub_img_dirs = os.listdir(img_dir)
    select_img_dic = {}
    for cls_name in select_cls:
        cls_dirs = [d for d in sub_img_dirs if cls_name in d and cls_name[0] == d[0]]
        select_img_dic[cls_name] = cls_dirs
    for n, dir_list in  select_img_dic.items():
        out_n_dir = os.path.join(out_dir, n)
        if not os.path.exists(out_n_dir):
            os.makedirs(out_n_dir)
        out_imgs = []
        for d in dir_list:
            d_pth = os.path.join(img_dir, d)
            imgs = os.listdir(d_pth)
            out_imgs += [os.path.join(d_pth, im) for im in imgs]
        random.shuffle(out_imgs)
        for i in range(min(len(out_imgs), count_per_cls)):
            im = cv2.imread(out_imgs[i])
            out_name = n + "_" + os.path.basename(out_imgs[i])
            cv2.imwrite(os.path.join(out_n_dir, out_name), im)


if __name__ == "__main__":
    img_dir = "H:/WorkSpace/20210126_ele_cad_project/Datasets/huabei_data/recognition_dataset"
    out_dir = "H:/WorkSpace/20210126_ele_cad_project/Datasets/huabei_data/pinNetDataset"
    if not os.path.exists(img_dir):
        print("paht not exist!")
        exit
    select_cls = ["dz0","jddz0", "kg0"]
    select_imgs(img_dir, out_dir, select_cls)