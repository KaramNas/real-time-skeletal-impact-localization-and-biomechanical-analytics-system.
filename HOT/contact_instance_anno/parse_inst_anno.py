import os
import json
import glob
from tqdm import tqdm
import numpy as np
import PIL.Image
from skimage.draw import polygon


new_part = ['Head', 'Chest', 'Back', 'LeftUpperArm', 'LeftForeArm', 'leftHand', 'rightUpperArm',
            'rightForeArm', 'RightHand', 'Butt', 'Hip', 'LeftThigh', 'LeftCalf', 'leftFoot',
            'rightThigh', 'rightCalf', 'rightFoot']
new_part_dict = {k.lower(): v+1 for v, k in enumerate(new_part)}
# {'Head': 1, 'Chest': 2, 'Back': 3, 'leftUpperArm,': 4, 'leftForeArm': 5, 'LeftHand': 6, 
# 'rightUpperArm': 7, 'rightForeArm': 8, 'rightHand': 9, 'Butt': 10, 'Hip': 11, 'leftThigh': 12,
# 'leftCalf': 13, 'leftFoot': 14, 'RightThigh': 15, 'rightCalf': 16, 'rightFoot': 17}


def get_concat_h(im1, im2):
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def parse_annotation(anno_path, img, vis=True):
    if not os.path.exists(anno_path):
        print(f'File not exists: {anno_path}')
        return None
    contact_inst = json.load(open(anno_path))
    height, width = contact_inst["imsize"]['height'], contact_inst["imsize"]['width']
    filename = contact_inst["filename"]

    for obj_info in contact_inst["object"]:
        contact_mask = np.zeros((height, width))
        idx = obj_info["id"]
        semantic_label = obj_info["semantic_label"]
        semantic_id  = obj_info["semantic_id"]
        poly = np.vstack([obj_info["polygon"]["x"], obj_info["polygon"]["y"]]).T

        print(poly.shape)
        rr, cc = polygon(poly[:, 1], poly[:, 0], contact_mask.shape)
        for v_id, _ in enumerate(rr):
            if rr[v_id] > height or rr[v_id] < 1:
                continue
            if cc[v_id] > width or cc[v_id] < 1:
                continue
            contact_mask[int(rr[v_id]), int(cc[v_id])] = 255 # semantic_id

        if vis:
            contact_mask = PIL.Image.fromarray(contact_mask)
            contact_mask = contact_mask.convert("RGB")
            total_img = PIL.Image.blend(contact_mask, img, 0.5)
            total_img.show(title=f"{filename}_{idx}_{semantic_label}")
    return None


def main():
    data_dir = 'path/to/instance/json'
    img_dir = 'path/to/all/images/training'

    list_json_files = sorted(glob.glob(data_dir + '/*.json'))

    for curr_file in tqdm(list_json_files):
        file_name = curr_file.split('/')[-1]
        img_name = file_name.split('.')[0]
        img = PIL.Image.open(os.path.join(img_dir, img_name + '.jpg'))
        parse_annotation(curr_file, img)


if __name__ == '__main__':
    main()
