import cv2
import sys
import argparse
import numpy as np
import os
import json

def gen_data_single(source_image, mask):
    img = cv2.imread(source_image)
    mask2 = np.zeros((mask.shape[0], mask.shape[1], 1))
    mask2 = mask[:, :, 0:1] + mask[:, :, 1:2] + mask[:, :, 2:]
    mask2[np.nonzero(mask2)] = 1
    img = img - img*mask2
    img = img + mask
    img = img[161:1208, 442:1489]
    resize_img = cv2.resize(img, (224, 224))     
    return resize_img

def gen_data(mask_file, dataset_path, output_path, img_size=(224, 224)):
    # root_path = "E:\\a2d2\\camera_lidar_semantic\\%s\\camera\\cam_front_center" % dataset_path
    mask = cv2.imread(mask_file)
    # mask = np.concatenate([mask, np.zeros((1208, 150, 3))],axis=1)
    # mask = mask[:, 150:, :]
    mask2 = np.zeros((mask.shape[0], mask.shape[1], 1))
    mask2 = mask[:, :, 0:1] + mask[:, :, 1:2] + mask[:, :, 2:]
    mask2[np.nonzero(mask2)] = 1
    # mask = cv2.cvtColor(mask, cv2.cvtColor)
    for d in os.listdir(dataset_path):
        if 'png' in d:
            img = cv2.imread(os.path.join(dataset_path, d))
            img = img - img*mask2
            img = img + mask
            img = img[161:1208, 442:1489]
            resize_img = cv2.resize(img, (img_size, img_size))
            image_json = d[:-4] + '.json'
            with open(os.path.join(dataset_path,image_json), 'r') as f:
                image_info = json.load(f)            
                timestamp = image_info["cam_tstamp"]
                # cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)
                cv2.imwrite(os.path.join(output_path, str(timestamp) + '.png'), resize_img)
                print(os.path.join(output_path, str(timestamp) + '.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mask", type=str)
    parser.add_argument("--object", type=str)
    parser.add_argument("--location", type=int)
    # parser.add_argument("--object", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    # parser.add_argument("--x_n", type=str)
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()
    if args.object == "pedestrian":
        mask = "../generators/OpenCV/person_50.png"
    elif args.object == "bicycle":
        mask = "../generators/OpenCV/bicycle_50.png"
    elif args.object == "vehicle":
        if args.location == 50:
            mask = "../generators/OpenCV/car_50.png"
        elif args.location == 30:
            mask = "../generators/OpenCV/car_30.png"
    elif args.object == "traffic_sign":
        mask = "../generators/OpenCV/speed_30.png"
    print(mask)
    gen_data(mask, args.dataset_path,args.output_path, args.img_size)