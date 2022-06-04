import time
from collections import OrderedDict

import pandas as pd
import yaml
import PySimpleGUI as sg
from PIL import Image
# from config import *
import os
import subprocess
import cv2
import torchvision.transforms as T
from data_a2d2 import A2D2MT, A2D2Diff
import json
import sys

sys.path.append("..")
# from data_a2d2 import A2D2Diff, A2D2MT
import importlib
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import copy
import matplotlib.image as mpimg
import numpy as np
from nlp_parser_new import Parser
from engine import Engine
from torchvision import transforms, utils

from create_compare_image import create_compare_image

THRESHOLD_LOW = 0.1
THRESHOLD_HIGH = 0.2
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def get_input_layout(model_names, dataset_names):
    input_layout = [
        [sg.Frame(layout=[
            [sg.Multiline(size=(52, 4), key="rule", default_text="If:\nThen:\nIf:\nThen:\n"), ],
        ],
            title='Rule to apply', relief=sg.RELIEF_SUNKEN)],

        [sg.Frame(layout=[
            # [sg.Text("Input pair (X_N1, X_N2):", size=(20, 1)), sg.Combo(("(Original image, Transformed image)", "(Transformed images 1 and 2)"), size=(30, 1), default_value="(Original image, Transformed image)", key="MRV_pair_type")],
            [sg.Text("Test Model:", size=(20, 1)),
             sg.Combo(model_names, size=(30, 1), default_value=model_names[0], key="MRV_MUT")],
            [sg.Text("Input dataset:", size=(20, 1)),
             sg.Combo(dataset_names, size=(30, 1), default_value=dataset_names[0], key="MRV_pair_data")],
            # [sg.Text("Inequation:", size=(20, 1)), sg.Combo(("decreaseRatio", "current", "deviation"), size=(20, 1), default_value="decreaseRatio", key="MRV_RF")],
            # [sg.Text("Range:", size=(20, 1)), sg.Input(key="MRV_Range_low", size=(11, 1)), sg.Text("to", size=(5,1)), sg.Input(key="MRV_range_high", size=(11, 1))],
        ],
            title='MR setting', relief=sg.RELIEF_SUNKEN)],

        [sg.Button("Generate test"), sg.Cancel()]
    ]
    return input_layout


def get_info(model_list, data_list):
    models = []
    datasets = []

    for model in model_list:
        models.append(model.name)

    for dataset in data_list:
        datasets.append(dataset.name)

    return models, datasets


def make_prediction_steer(model, phase=2):

    with torch.no_grad():
        pred_x1 = []
        pred_x2 = []
        pred_x3 = []
        image_list = os.listdir("test_images/x_n1")
        image_list.sort()
        image_list = image_list[1:]
        for image_name in image_list:
            x_n1_image = cv2.imread(os.path.join("test_images/x_n1", image_name))
            # x_n1_image = cv2.resize(x_n1_image, (, 160))
            x_n1_image = cv2.cvtColor(x_n1_image, cv2.COLOR_BGR2RGB)
            x_n1_image = np.transpose(x_n1_image, (-1, 0, 1))
            x_n1_image = x_n1_image / 255
            x_n1_image = torch.from_numpy(x_n1_image).unsqueeze(0)
            x_n1_image = x_n1_image.type(torch.FloatTensor).to(device)

            x_n2_image = cv2.imread(os.path.join("test_images/x_n2", image_name))
            # x_n2_image = cv2.resize(x_n2_image, (320, 160))
            x_n2_image = cv2.cvtColor(x_n2_image, cv2.COLOR_BGR2RGB)
            x_n2_image = np.transpose(x_n2_image, (-1, 0, 1))
            x_n2_image = x_n2_image / 255
            x_n2_image = torch.from_numpy(x_n2_image).unsqueeze(0)
            x_n2_image = x_n2_image.type(torch.FloatTensor).to(device)

            pred_x1.append(model(x_n1_image).item())
            pred_x2.append(model(x_n2_image).item())

            if phase > 2:
                x_n3_image = cv2.imread(os.path.join("test_images/x_n3", image_name))
                # x_n2_image = cv2.resize(x_n2_image, (320, 160))
                x_n3_image = cv2.cvtColor(x_n3_image, cv2.COLOR_BGR2RGB)
                x_n3_image = np.transpose(x_n3_image, (-1, 0, 1))
                x_n3_image = x_n3_image / 255
                x_n3_image = torch.from_numpy(x_n3_image).unsqueeze(0)
                x_n3_image = x_n3_image.type(torch.FloatTensor).to(device)   
                pred_x3.append(model(x_n3_image).item())

        if phase > 2:
            return pred_x1, pred_x2, pred_x3
        else:
            return pred_x1, pred_x2


def make_predictions_speed(model, x_n1, x_n2):
    transform = T.Resize(size = (160,320))
    with torch.no_grad():
        bg_speed = []
        label = []
        source_pred = []
        follow_up_pred = []
        # dataloader
        for i in range(len(x_n1)):
            source_images = x_n1[i][0]
            source_bg_speed = x_n1[i][1]
            source_label = x_n1[i][2]

            follow_up_images = x_n2[i][0]
            follow_up_bg_speed = x_n2[i][1]
            follow_up_label = x_n2[i][2]
            label.append(source_label)

            source_images = source_images.type(torch.FloatTensor)
            follow_up_images = follow_up_images.type(torch.FloatTensor)
            source_speed = torch.tensor(source_bg_speed).type(torch.FloatTensor)
            follow_speed = torch.tensor(follow_up_bg_speed).type(torch.FloatTensor)
            # print(source_speed, follow_speed)
            source_input = (source_images.unsqueeze(0).to(device), source_speed.unsqueeze(0).to(device))
            follow_up_input = (follow_up_images.unsqueeze(0).to(device), follow_speed.unsqueeze(0).to(device))
            source_output = model(source_input)
            follow_up_output = model(follow_up_input)
            bg_speed.append(source_speed.mean().item())
            source_pred.append(source_output.item())
            follow_up_pred.append(follow_up_output.item())
        return source_pred, follow_up_pred


def resize_img(dataset, x_n):
    if dataset.name == "A2D2":
        for d in os.listdir(dataset.path):
            if 'png' in d:
                img = cv2.imread(os.path.join(dataset.path, d))
                img = img[161:1208, 442:1489]
                resize_img = cv2.resize(img, (int(dataset.img_size), int(dataset.img_size)))
                image_json = d[:-4] + '.json'
                with open(os.path.join(dataset.path, image_json), 'r') as f:
                    image_info = json.load(f)
                    timestamp = image_info["cam_tstamp"]
                    # cv2.imwrite(os.path.join(root_path, "camera_resize", folder, str(timestamp) + '.png'), resize_img)
                    cv2.imwrite(os.path.join("../test_images/" + x_n, str(timestamp) + '.png'), resize_img)
    elif dataset.name == "Cityscapes":
        for d in os.listdir(dataset.path):
            if 'png' in d:
                img = cv2.imread(os.path.join(dataset.path, d))
                img = img[0:852, 384:1520]
                resize_img = cv2.resize(img, (int(dataset.img_size), int(dataset.img_size)))
                cv2.imwrite(os.path.join("../test_images/" + x_n, d), resize_img)

def get_output_layout(mt_result):
    img_list = os.listdir("test_images/" + mt_result[2])
    img_list.sort()
    img_list = [img for img in img_list if img[0] == '1']



    Image.open(os.path.join("test_images/" + mt_result[2], img_list[1])).save("result_sample_0.png")
    # Image.open(os.path.join("test_images/" + mt_result[2], img_list[-1])).save("resule_sample_{}.png".format(i))

    out_layout = [
        [sg.Text(mt_result[0], font=('Helvetica', 18))],
        [sg.Text("Created MR: " + ';'.join(mt_result[1]), font=('Helvetica', 18))],
        [sg.Frame(layout=[
            # [sg.Image('original_1.png'),
            #  sg.Image('original_2.png'),
            #  sg.Image('original_3.png')]
            [sg.Image("result_sample_0.png")]
        ],
            title='Sample of result', relief=sg.RELIEF_SUNKEN, font=('Helvetica', 14))],
        # [sg.Frame(layout=[
        #     # [sg.Image('result_1.png'),
        #     #  sg.Image('result_2.png'),
        #     #  sg.Image('result_3.png')]
        #     [sg.Image("result_sample_1.png")]
        #
        # ],
        #
        #     title='Generated graph', relief=sg.RELIEF_SUNKEN)],
        [sg.Text("All result folder: " + os.path.join("test_images/" + mt_result[2]), font=('Helvetica', 18))],
        # [sg.Image('../source_datasets/orginal/1.jpg'), sg.Image('../follow_up_datasets/rainy/2.jpg')],
        [sg.OK(key="result_ok")]
    ]
    return out_layout

def clear_compare_images(compare_image_folder):
    img_list = os.listdir(os.path.join("test_images", compare_image_folder))
    for img_name in img_list:
        if 'png' in img_name:
            os.remove(os.path.join("test_images", compare_image_folder, img_name))

def clear_test_images(phase=1):
    if phase == 1:
        img_list = os.listdir("test_images/x_n1")
        for img_name in img_list:
            if 'png' in img_name:
                os.remove(os.path.join("test_images/x_n1", img_name))

    img_list = os.listdir("test_images/x_n" + str(phase+1))
    for img_name in img_list:
        if 'png' in img_name:
            os.remove(os.path.join("test_images/x_n" + str(phase+1), img_name))

def pipeline(values, generator_list, model_list, data_list,
             compare_image_folder=None, gen_image=True):

    # parser = NLP_Parser(location_list, object_list, transformation_list)
    # parser = NLP_parser()
    parser = Parser()
    rule = values["rule"]
    target_transformation, target_object, target_parameter, MR_relation, work_engines = parser.rule_parse(rule)
    print(MR_relation)
    if None in work_engines:
        can_support = False
    else:
        can_support = True
    
    # for transformation, object, parameter in zip(target_transformation, target_object, target_parameter):
    #     for support_transformation in transformation_list:
    #         if support_transformation.name == transformation and (parameter in support_transformation.support_parameter or
    #          (parameter == None and support_transformation.parameter_name ==None)):
    #              for s_object in support_transformation.support_object: 
    #                 if s_object in object:
    #                     can_support = True
    #                     break

    #         if can_support:
    #             break
    #     if can_support:
    #         break

    if not can_support:
        return False, "No transformation engine supports the input rule", None, None, None
    selected_dataset = None
    for dataset_name, dataset in data_list.items():
        if dataset_name == values["MRV_pair_data"]:
            selected_dataset = dataset
            break
    selected_model = None
    for model_name, model in model_list.items():
        if model_name == values["MRV_MUT"]:
            selected_model = model

    if gen_image and can_support:
        gen_start = time.time()
        work_engine = Engine()
        if len(target_transformation) == 1:
            work_engine.set_generator(work_engines[0], generator_list)
            # if not is_set:
                # return False, "No transformation engine supports the input rule", None, None, None
            clear_test_images()
            work_engine.apply_transform(target_object[0], target_transformation[0], target_parameter[0], selected_dataset['path'],
                                    selected_dataset['semantic_path'], 'test_images')

        else:
            for i, transformation in enumerate(target_transformation):
                if i == 0:
                    work_engine.set_generator(work_engines[i], generator_list)
                    clear_test_images(phase=i+1)
                    work_engine.apply_transform(target_object[i], target_transformation[i],target_parameter[i], selected_dataset['path'],
                                            selected_dataset['semantic_path'], 'test_images', phase=i+1)
                else:
                    work_engine.set_generator(target_object[i], target_transformation[i], target_parameter[i], generator_list)
                    clear_test_images(phase=i+1)

                    work_engine.apply_transform(target_object[i], target_transformation[i],target_parameter, selected_dataset['path'],
                                        selected_dataset['semantic_path'], 'test_images', phase=i+1)

            # sync generated images in x_n2 and x_n3
            common_image_list = [img_name for img_name in os.listdir('test_images/x_n2') if img_name
                                 in os.listdir('test_images/x_n3')]
            common_image_list.sort()
            # common_image_list = common_image_list[1:]

            x_n2_list = os.listdir('test_images/x_n2')
            for img_name in x_n2_list:
                if img_name not in common_image_list:
                    os.remove(os.path.join("test_images/x_n2", img_name))

            x_n3_list = os.listdir('test_images/x_n3')
            for img_name in x_n3_list:
                if img_name not in common_image_list:
                    os.remove(os.path.join("test_images/x_n3", img_name))
        gen_end = time.time()
        print("Time cost of image generation: ", gen_end - gen_start)
    else:
        if len(target_transformation) > 1:
            common_image_list = [img_name for img_name in os.listdir('test_images/x_n2') if img_name
                                 in os.listdir('test_images/x_n3')]
            common_image_list.sort()
            # common_image_list = common_image_list[1:]
    # load driving models
    driving_model_module = __import__(selected_model['class_file'])
    driving_model = getattr(driving_model_module, selected_model['class_name'])()
    if selected_model['distributed'] == "1":
        driving_model = nn.DataParallel(driving_model, device_ids=[0])
        driving_model.load_state_dict(torch.load(selected_model['path']))
    else:
        state_dict_parallel = torch.load(selected_model['path'])
        state_dict_single = OrderedDict([(k[7:], v) for (k, v) in state_dict_parallel.items()])
        driving_model.load_state_dict(state_dict_single)
    driving_model = driving_model.to(device)
    driving_model.eval()


    # make predictions for mt sets
    if selected_dataset['name'] == "A2D2":
        split_path = selected_dataset['path'].split(os.sep)
        root_path = split_path[0]
        for i in range(1, split_path.index('a2d2') + 1):
            root_path += os.sep
            root_path += split_path[i]

        if 'steer' in selected_model['name']:
            pred_x1, pred_x2 = make_prediction_steer(driving_model)
            label_type = 'Steer angle'
        else:
            label_type = 'Speed'
            root_path = os.path.join('/', *split_path[:split_path.index('a2d2') + 1])[0:]
            mt_set_x1 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=False,
                               transform=transforms.ToTensor())
            mt_set_x2 = A2D2MT(root_path, mt_path="test_images/x_n2", get_mt=True,
                               transform=transforms.ToTensor())
            pred_x1, pred_x2 = make_predictions_speed(driving_model, mt_set_x1, mt_set_x2)

            if len(MR_relation) > 1:
                mt_set_x3 = A2D2MT(root_path, mt_path="test_images/x_n3", get_mt=True, transform=transforms.ToTensor())
                pred_x1, pred_x3 = make_predictions_speed(driving_model, mt_set_x1, mt_set_x3)

    # create metamorphic relation
    evaluate_start = time.time()
    if len(MR_relation) == 1:
        relation_function = lambda x1, x2: eval(MR_relation[0])
        total = len(pred_x1)
        violation = 0
        violation_record = []
        for (y1, y2) in zip(pred_x1, pred_x2):
            if not relation_function(y1, y2):
                violation += 1
                violation_record.append(1)
            else:
                violation_record.append(0)

        print(violation, total)
        img_list = os.listdir("test_images/x_n1")
        img_list.sort()
        img_list = img_list[1:] # remove .gitignore
        df = pd.DataFrame({"Image name": img_list, "Original pred": pred_x1, "Transformed pred": pred_x2})
        df.to_csv(compare_image_folder + ".csv")
        if os.path.exists(compare_image_folder):
            clear_compare_images(compare_image_folder)
        create_compare_image((pred_x1, pred_x2), img_list, violation_record, compare_image_folder,
                             label_type, False)



    else:
        relation_function_1 = lambda x1, x2: eval(MR_relation[0])
        relation_function_2 = lambda x2, x3: eval(MR_relation[1])
        total = len(pred_x1)
        violation = 0
        violation_record = []

        for i in range(total):
            if not relation_function_1(pred_x1[i], pred_x2[i]) or not relation_function_2(pred_x2[i], pred_x3[i]):
                violation += 1
                violation_record.append(1)
            else:
                violation_record.append(0)

        print(violation, total)

        df = pd.DataFrame({"Image name": common_image_list, "Original pred": pred_x1, "Transformed pred_1": pred_x2,
                           "Transformed pred_2": pred_x3, })
        df.to_csv(compare_image_folder + ".csv")
        if os.path.exists(compare_image_folder):
            clear_compare_images(compare_image_folder)
        create_compare_image((pred_x1, pred_x2, pred_x3), common_image_list, violation_record, compare_image_folder,
                             label_type, False)

    evaluate_end = time.time()
    print("Time cost of evaluation: ", evaluate_end - evaluate_start)
    del driving_model
    torch.cuda.empty_cache()
    return True, "test finished", violation, total, MR_relation


def load_config(config_file):
    with open("config/{}".format(config_file), "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    return config



if __name__ == "__main__":
    if os.environ.get('DISPLAY', '') == '':
        print('no display found. Using :0.0')
        os.environ.__setitem__('DISPLAY', ':0.0')
    # create_compare_image_steer([[0.5, 0.3]])
    # location_list, object_list, transformation_list, generator_list, model_list, data_list = parse_new()
    # models, datasets = get_info(model_list, data_list)
    config = load_config('configuration.yaml')
    generator_list = config['engine']
    models = config['model']
    datasets = config['dataset']

    # selected_trans = []
    # trans_for_x1 = []
    # trans_for_x2 = []
    input_window = sg.Window('Rule-based Metamorphic Testing').Layout(get_input_layout(list(models.keys()), list(datasets.keys())))
    # current_trans = TransformationObject([None, None, None, None])

    while True:
        button, values = input_window.Read()
        print(button, values)
        if button in (None, 'Exit'):
            break

        elif button == "Generate test":
            success, info, violation, total, MR_relation = pipeline(values, generator_list,
                                        models, datasets, compare_image_folder="current_rule", gen_image=True)
            if success:
            # display result window
                mt_result = "{}  violations were found from {} test cases.".format(violation, total)
                result_param = [mt_result, MR_relation, 'current_rule']
                selected_dataset = None
                selected_model = None
                trans_for_x1 = []
                trans_for_x2 = []
                input_window.close()
                out_window = sg.Window('show').Layout(get_output_layout(result_param))
                button, values = out_window.Read()
                if button == "result_ok":
                    out_window.close()

                    os.remove("result_sample_0.png")
                    input_window = sg.Window('Rule-based Metamorphic Testing').Layout(get_input_layout(list(models.keys()), list(datasets.keys())))
            else:
                print(info)