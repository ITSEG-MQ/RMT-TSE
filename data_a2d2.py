import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, transform
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms  
import pandas as pd
import os 
import torch
from PIL import Image
import os
import json
import cv2
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
import time
from datetime import datetime
import shutil

class A2D23D(Dataset):
    def __init__(self, phase="train",sequence=3, transform=None, image_size=(224, 224), root_path='.'):
        self.root_path = root_path
        self.phase = phase
        self.transform = transform
        self.image_size = image_size
        self.sequence=sequence
        self.image_list, self.label_list = self.get_data_list() 

    def get_data_list(self):
        label_root_folder = os.path.join(self.root_path, "bus_data")
        label_folders = []
        for d in os.listdir(label_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "bus_data", d)):
                label_folders.append(d)
        image_root_folder =  os.path.join(self.root_path, "camera_lidar_semantic")
        image_folders = []
        for d in os.listdir(image_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "camera_lidar_semantic", d)):
                image_folders.append(d)
        public_folders = [d for d in label_folders if d in image_folders]
        dataset_folders = None
        if self.phase == "train":
            dataset_folders = public_folders[1:-1]
        elif self.phase == "validation":
            dataset_folders = public_folders[-1:]
        else:
            dataset_folders = public_folders[0:1]

        image_list = []
        label_list = []
        for i in range(len(dataset_folders)):
            folder = dataset_folders[i] 
            folder_images = {}
            labels = {}
            for d in os.listdir(os.path.join(self.root_path, "camera_resize", folder)):
                folder_images[d[:-4]] = os.path.join(self.root_path, "camera_resize", folder, d)

            bus_data_file = os.listdir(os.path.join(self.root_path, "bus_data", folder, "bus"))[0]
            with open(os.path.join(self.root_path, "bus_data", folder, "bus", bus_data_file), 'r') as f:
                bus_data = json.load(f)
                for data in bus_data:
                    if 'vehicle_speed' in data['flexray']:
                        timestamp = data['timestamp']   
                        vehicle_speed = sum(data['flexray']['vehicle_speed']['values']) / len(data['flexray']['vehicle_speed']['values'])
                        if vehicle_speed > 5:
                            labels[str(timestamp)] = vehicle_speed
            
            for timestamp in labels.keys():
                if timestamp in folder_images.keys():
                    image_list.append(folder_images[timestamp])
                    label_list.append(labels[timestamp])
        # print(len(image_list), len(label_list))        
        image_list = np.array(image_list)
        label_list = np.array(label_list)
        return image_list, label_list    

    def __len__(self):
        return len(self.image_list) - self.sequence + 1
    
    def __getitem__(self, idx):
        imgs = []
        for i in range(self.sequence):
            img = cv2.imread(self.image_list[idx+i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)   

        label = self.label_list[idx + self.sequence - 1]

        if self.transform:
            imgs_transform = [self.transform(img) for img in imgs]

            imgs = torch.stack(imgs_transform, dim=0)
        else:
            imgs = np.stack(imgs_transform, axis=0)
        
        return imgs, label        

def count_images(folders):
    count = 0
    for folder in folders:
        cur_dir = os.path.join("E:\\a2d2\\camera_resize", folder)
        count += len(os.listdir(cur_dir))
    return count

class A2D2Diff(Dataset):
    def __init__(self, phase="train", transform=None, image_size=(160, 320), root_path='.'):
        self.root_path = root_path
        self.phase = phase
        self.transform = transform
        self.image_size = image_size
        self.image_list, self.speed_list, self.label_list = self.get_data_list()

    def get_data_list(self):
        label_root_folder = os.path.join(self.root_path, "bus_data")
        label_folders = []
        for d in os.listdir(label_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "bus_data", d)):
                label_folders.append(d)
        image_root_folder =  os.path.join(self.root_path, "camera_lidar_semantic")
        image_folders = []
        for d in os.listdir(image_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "camera_lidar_semantic", d)):
                image_folders.append(d)
        public_folders = [d for d in label_folders if d in image_folders]

        dataset_folders = None
        if self.phase == "train":
            dataset_folders = public_folders[1:-1]
        elif self.phase == "validation":
            dataset_folders = public_folders[-1:]
        else:
            dataset_folders = public_folders[:1]
        # print(dataset_folders)
        # print(count_images(public_folders[1:14] + public_folders[15:]))
        # print(count_images(public_folders[0:1]))
        # print(count_images(public_folders[14:15]))
        image_list = []
        speed_list = []
        label_list = []
        for i in range(len(dataset_folders)):
            folder = dataset_folders[i] 
            folder_images = {}
            labels = {}
            speeds = {}
            for d in os.listdir(os.path.join(self.root_path, "camera_resize", folder)):
                folder_images[d[:-4]] = os.path.join(self.root_path, "camera_resize", folder, d)
            # folder_images = OrderedDict(sorted(folder_images.items()))
            bus_data_file = os.listdir(os.path.join(self.root_path, "bus_data", folder, "bus"))[0]
            with open(os.path.join(self.root_path, "bus_data", folder, "bus", bus_data_file), 'r') as f:
                bus_data = json.load(f)
                for i, data in enumerate(bus_data):
                    if 'vehicle_speed' in data['flexray']:
                        timestamp = data['timestamp']   
                        vehicle_speed = sum(data['flexray']['vehicle_speed']['values']) / len(data['flexray']['vehicle_speed']['values'])
                        labels[str(timestamp)] = vehicle_speed
                        speeds[str(timestamp)] = np.array(data['flexray']['vehicle_speed']['values'])
            sorted_timestamp = list(folder_images.keys())
            sorted_timestamp.sort()
            images = [folder_images[t] for t in sorted_timestamp if t in labels.keys()]
            labels = [labels[t] for t in sorted_timestamp if t in labels.keys()]
            speeds = [speeds[t] for t in sorted_timestamp if t in speeds.keys()]
            # if (len(folder_images) == len(labels)):
                # folder_images = list(folder_images.values())
                # speeds = list(labels.values())
            for i in range(1, len(images)):
                # image_list.append(images[i])
                # if i == 0:
                #     image_list.append([images[0], images[0]])
                #     label_list.append(0)
                #     speed_list.append(speeds[0])
                # else:
                image_list.append([images[i-1], images[i]])
                label_list.append(labels[i])
                speed_list.append(speeds[i - 1])
        # print(len(image_list), len(label_list))        
        image_list = np.array(image_list)
        label_list = np.array(label_list)
        speed_list = np.array(speed_list)
        # speed_list = np.array(speed_list)
        return image_list,speed_list, label_list
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # img = io.imread(self.image_list[idx])
        imgs = [cv2.imread(self.image_list[idx][i]) for i in range(2)]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        speed = self.speed_list[idx]
        # img = img[128:, :]
        # print(img.shape)
        label = self.label_list[idx]
        # for img in imgs:
        #     img = img[248:,:]
        #     img = cv2.resize(img, self.image_size)

        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        return imgs, speed, label


class A2D2MT(Dataset):
    def __init__(self, root_path, mt_path, get_mt=True, transform=None):
        self.root_path = root_path
        self.mt_path = mt_path
        self.get_mt = get_mt
        self.transform = transform
        # self.data = A2D2Diff(phase="validation",transform=transforms.ToTensor(), root_path=root_path)
        self.original_data = self.get_original_data()
        self.mt_image_list = os.listdir(mt_path)
        self.mt_image_list.sort()
        # self.mt_image_list = self.mt_image_list[1:] # remove .gitignore file

    def get_original_data(self):
        label_root_folder = os.path.join(self.root_path, "bus_data")
        label_folders = []
        for d in os.listdir(label_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "bus_data", d)):
                label_folders.append(d)
        image_root_folder =  os.path.join(self.root_path, "camera_lidar_semantic")
        image_folders = []
        for d in os.listdir(image_root_folder):
            if os.path.isdir(os.path.join(self.root_path, "camera_lidar_semantic", d)):
                image_folders.append(d)
        public_folders = [d for d in label_folders if d in image_folders]

        dataset_folders = public_folders[:1]

        for i in range(len(dataset_folders)):
            folder = dataset_folders[i]
            folder_images = {}
            labels = {}
            speeds = {}
            for d in os.listdir(os.path.join(self.root_path, "camera_resize", folder)):
                folder_images[d[:-4]] = os.path.join(self.root_path, "camera_resize", folder, d)
            # folder_images = OrderedDict(sorted(folder_images.items()))
            bus_data_file = os.listdir(os.path.join(self.root_path, "bus_data", folder, "bus"))[0]
            with open(os.path.join(self.root_path, "bus_data", folder, "bus", bus_data_file), 'r') as f:
                bus_data = json.load(f)
                for i, data in enumerate(bus_data):
                    if 'vehicle_speed' in data['flexray']:
                        timestamp = data['timestamp']
                        vehicle_speed = sum(data['flexray']['vehicle_speed']['values']) / len(data['flexray']['vehicle_speed']['values'])
                        labels[str(timestamp)] = vehicle_speed
                        speeds[str(timestamp)] = np.array(data['flexray']['vehicle_speed']['values'])
            sorted_timestamp = list(folder_images.keys())
            sorted_timestamp.sort()
            images = [folder_images[t] for t in sorted_timestamp if t in labels.keys()]
            labels = [labels[t] for t in sorted_timestamp if t in labels.keys()]
            speeds = [speeds[t] for t in sorted_timestamp if t in speeds.keys()]

        return images, speeds, labels

    def __len__(self):

        return len(self.mt_image_list)

    def __getitem__(self, idx):
        mt_image_name = self.mt_image_list[idx]
        original_image_list = [image_name.split('/')[-1] for image_name in self.original_data[0]]
        mt_pos = original_image_list.index(mt_image_name)
        images = []
        if mt_pos != 0:
            images.append(cv2.imread(self.original_data[0][mt_pos - 1]))
            speed = self.original_data[1][mt_pos - 1]
            label = self.original_data[2][mt_pos - 1]
        else:
            images.append(cv2.imread(self.original_data[0][0]))
            speed = self.original_data[1][0]
            label = self.original_data[2][0]
        if self.get_mt:
            images.append(cv2.imread(os.path.join(self.mt_path, mt_image_name)))
        else:
            images.append(cv2.imread(self.original_data[0][mt_pos]))

        # imgs = [cv2.imread(self.image_list[idx][i]) for i in range(2)]
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]


        # for img in images:
        #     if img.shape != self.data.image_size:
        #         img = img[248:,:]
        #         img = cv2.resize(img, self.data.image_size)

        if self.transform:
            imgs = [self.transform(img) for img in images]
        imgs = torch.stack(imgs, dim=0)
        return imgs, speed, label

# class A2D2MT(Dataset):
#     def __init__(self, root_path, source_path=None, mt_path=None, mode="compare", folder="20181108_091945"):
#         self.mt_path = mt_path
#         self.root_path = root_path
#         self.source_path = source_path
#         self.mode = mode
#         self.timestamp_list = []
#         self.image_list, self.speed_list, self.label_list = self.get_data_list(folder)
#
#     def __len__(self):
#         return len(self.image_list)
#     def get_data_list(self, folder):
#         image_list = []
#         speed_list = []
#         label_list = []
#         # mt_folder = self.mt_path
#         folder_images = {}
#         labels = {}
#         speeds = {}
#         for d in os.listdir(os.path.join(self.root_path, "camera_resize", folder)):
#             folder_images[d[:-4]] = d
#
#         bus_data_file = os.listdir(os.path.join(self.root_path, "bus_data", folder, "bus"))[0]
#         with open(os.path.join(self.root_path, "bus_data", folder, "bus", bus_data_file), 'r') as f:
#             bus_data = json.load(f)
#             for i, data in enumerate(bus_data):
#                 if 'vehicle_speed' in data['flexray']:
#                     timestamp = data['timestamp']
#                     self.timestamp_list.append(timestamp)
#                     vehicle_speed = sum(data['flexray']['vehicle_speed']['values']) / len(data['flexray']['vehicle_speed']['values'])
#                     labels[str(timestamp)] = vehicle_speed
#                     speeds[str(timestamp)] = np.array(data['flexray']['vehicle_speed']['values'])
#         images = [folder_images[t] for t in folder_images.keys() if t in labels.keys()]
#         labels = [labels[t] for  t in folder_images.keys() if t in labels.keys()]
#         speeds = [speeds[t] for t in folder_images.keys() if t in speeds.keys()]
#         # if (len(folder_images) == len(labels)):
#             # folder_images = list(folder_images.values())
#             # speeds = list(labels.values())
#         for i in range(1, len(images)):
#             # image_list.append(images[i])
#             if self.mode == "compare":
#                 if self.source_path == None:
#                     image_list.append([os.path.join(self.root_path, "camera_resize", folder, images[i-1]),  os.path.join(self.mt_path, images[i])])
#                 else:
#                     image_list.append([os.path.join(self.source_path, images[i-1]),  os.path.join(self.mt_path, images[i])])
#
#             else:
#                 image_list.append([os.path.join(self.mt_path, images[i - 1]),  os.path.join(self.mt_path, images[i])])
#             label_list.append(labels[i])
#             speed_list.append(speeds[i - 1])
#         # print(len(image_list), len(label_list))
#         # image_list = np.array(image_list)
#         label_list = np.array(label_list)
#         speed_list = np.array(speed_list)
#         # speed_list = np.array(speed_list)
#         return image_list,speed_list, label_list
#
#     def __getitem__(self, idx):
#         # img = io.imread(self.image_list[idx])
#         imgs = [cv2.imread(self.image_list[idx][i]) for i in range(2)]
#         imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
#
#         speed = self.speed_list[idx]
#         # img = img[128:, :]
#         # print(img.shape)
#         label = self.label_list[idx]
#
#         # img = cv2.resize(img, self.image_size)
#
#         # if self.transform:
#         imgs = [transforms.ToTensor()(img) for img in imgs]
#         imgs = torch.stack(imgs, dim=0)
#         return imgs, speed, label

def gen_resize_data():
    root_path = "E:\\a2d2"
    label_root_folder = os.path.join(root_path, "bus_data")
    label_folders = []
    for d in os.listdir(label_root_folder):
        if os.path.isdir(os.path.join(root_path, "bus_data", d)):
            label_folders.append(d)
    image_root_folder =  os.path.join(root_path, "camera_lidar_semantic")
    image_folders = []
    for d in os.listdir(image_root_folder):
        if os.path.isdir(os.path.join(root_path, "camera_lidar_semantic", d)):
            image_folders.append(d)
    public_folders = [d for d in label_folders if d in image_folders]
    for folder in public_folders:
        if not os.path.exists(os.path.join(root_path, "camera_resize", folder)):
            os.mkdir(os.path.join(root_path, "camera_resize", folder))
        for d in os.listdir(os.path.join(root_path, "camera_lidar_semantic", folder, "camera", "cam_front_center")):
            if 'png' in d:
                img = cv2.imread(os.path.join(root_path, "camera_lidar_semantic", folder, "camera", "cam_front_center",d))
                img = img[161:1208, 442:1489]
                resize_img = cv2.resize(img, (224, 224))
                # resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                image_json = d[:-4] + '.json'
                with open(os.path.join(root_path, "camera_lidar_semantic", folder, "camera", "cam_front_center",image_json), 'r') as f:
                    image_info = json.load(f)            
                    timestamp = image_info["cam_tstamp"]
                    cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)


def gen_data(mask_file):
    if not os.path.exists(mask_file[:-4]):
        os.mkdir(mask_file[:-4])
    root_path = "E:\\a2d2\\camera_lidar_semantic\\20181108_091945\\camera\\cam_front_center"
    mask = cv2.imread(mask_file)
    # mask = np.concatenate([mask, np.zeros((1208, 150, 3))],axis=1)
    # mask = mask[:, 150:, :]
    mask2 = np.zeros((mask.shape[0], mask.shape[1], 1))
    mask2 = mask[:, :, 0:1] + mask[:, :, 1:2] + mask[:, :, 2:]
    mask2[np.nonzero(mask2)] = 1
    # mask = cv2.cvtColor(mask, cv2.cvtColor)
    for d in os.listdir(root_path):
        if 'png' in d:
            img = cv2.imread(os.path.join(root_path, d))
            img = img - img*mask2
            img = img + mask
            img = img[161:1208, 442:1489]
            resize_img = cv2.resize(img, (224, 224))
            image_json = d[:-4] + '.json'
            with open(os.path.join("..", "camera_lidar_semantic", "20181108_091945", "camera", "cam_front_center",image_json), 'r') as f:
                image_info = json.load(f)            
                timestamp = image_info["cam_tstamp"]
                # cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)

                cv2.imwrite(os.path.join(mask_file[:-4], str(timestamp) + '.png'), resize_img)
def get_hex(img, x, y):
    pixel = img[x, y]
    return ('#%02x%02x%02x' % (pixel[2], pixel[1], pixel[0]))

def check_data(filtered_path, type_="vehcile"):
    if not os.path.exists(filtered_path):
        os.mkdir(filtered_path)
    data_path = "E:\\a2d2\\camera_lidar_semantic\\20181108_091945\\camera\\cam_front_center"
    semantic_path = "E:\\a2d2\\camera_lidar_semantic\\20181108_091945\label\\cam_front_center"
    resize_data_path = "E:\\a2d2\\camera_resize\\20181108_091945"
    car_hex = ["#ff8000", "#c88000", "#968000", "#00ff00", "#00c800", "#009600", "#0080ff", "#ff0000", "#c80000", "#960000", "#800000"]
    # filtered_path = "E:\\a2d2\\filtered_image3"

    if "car" in type_:
        raw_mask = cv2.imread(type_ + ".png")
    else:
        raw_mask = cv2.imread(type_ + ".png")
        mask_pixels = np.array(np.nonzero(raw_mask)) # dim0: height dim1:width
        height_start = min(mask_pixels[0])
        obj_height = max(mask_pixels[0]) - min(mask_pixels[0])
        width_start = min(mask_pixels[1])
        obj_width = max(mask_pixels[1]) - min(mask_pixels[1])


    for d in os.listdir(semantic_path):
        semantic_img = cv2.imread(os.path.join(semantic_path, d))
        if  "car" in type_:
            check_area = semantic_img[740:865,950:1225]
        else :
            check_area = semantic_img[height_start:height_start+obj_height, width_start:width_start+obj_width]
        flag = False
        for i in range(check_area.shape[0]):
            for j in range(check_area.shape[1]):
                pixel_hex = get_hex(check_area, i, j)
                if pixel_hex in car_hex:
                    flag = True
                    break
        
        if not flag:
            image_name = d.replace("label", "camera")
            image_json = image_name[:-4] + '.json'
            with open(os.path.join("..", "camera_lidar_semantic", "20181108_091945", "camera", "cam_front_center",image_json), 'r') as f:
                image_info = json.load(f)            
                timestamp = image_info["cam_tstamp"]
                gen_img = gen_data_single(os.path.join("..", "camera_lidar_semantic", "20181108_091945", "camera", "cam_front_center", image_name), raw_mask)
                cv2.imwrite(os.path.join(filtered_path, str(timestamp) + '.png'), gen_img)
                # shutil.copy(os.path.join(resize_data_path, str(timestamp) + '.png'), os.path.join(filtered_path, str(timestamp) + '.png'))
def check_speed():
    path = "E:\\a2d2\\bus_data"
    for d in os.listdir(path):
        bus_json = os.path.join(path, d, "bus", d[:8] + d[9:] + "_bus_signals.json")
        with open(bus_json, 'r') as f:
            min_speed = 200
            max_speed = 0
            bus_data = json.load(f)
            for data in bus_data:
                if 'vehicle_speed' in data['flexray']:
                    current_speeds = data['flexray']['vehicle_speed']['values']
                    if min(current_speeds) < min_speed:
                        min_speed = min(current_speeds)
                    if max(current_speeds) > max_speed:
                        max_speed = max(current_speeds)
            # print(d, min_speed, max_speed)

if __name__ == "__main__":
    # mask = "bicycle_50.png"
    # gen_data(mask)
    check_data("../filter_add_person_50", "person_50")
    # check_speed()
    # train_composed = transforms.Compose([transforms.ToTensor()])
    
    # dataset = A2D23D(phase="test",root_path="..", transform=train_composed, sequence=5)
    # dataset = A2D2Diff(phase="test",root_path="..")
    # train_generator = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    # print(datetime.now())
    # for step, sample_batched in enumerate(train_generator):
    #     print (sample_batched[0].shape)
    #     print(datetime.now())
    # print(len(dataset))
    # print(dataset[100])
    # plt.imshow(dataset[0][0])
    # plt.show()
    # gen_resize_data()

    # dataset = A2D2MT(root_path="..", source_path="car_100", mt_path="car_100", mode="self")
    # plt.imshow(dataset[4][0][0])
    # plt.show()
    # plt.imshow(dataset[4][0][1])
    # plt.show()

    # check_data("E:\\a2d2\\filtered_image_car_50", type_="car_30")
