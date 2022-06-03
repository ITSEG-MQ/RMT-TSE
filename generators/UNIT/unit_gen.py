from __future__ import print_function

import json
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
# os.chdir(dir_path)
sys.path.insert(0, dir_path)
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import torch
from torchvision import transforms
from PIL import Image


def day2rain(dataset_path, output_path):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    config_file = dir_path + '/configs/unit_day2rain.yaml'
    config = get_config(config_file)
    trainer = UNIT_Trainer(config)
    trainer.cuda()
    trainer.eval()

    # state_dict = torch.load(dir_path + '/models/gen.pt')
    state_dict = torch.load('./models/generator_models/UNIT/gen_00030000.pt')

    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])


    encode = trainer.gen_a.encode
    style_encode = trainer.gen_b.encode
    decode = trainer.gen_b.decode
    new_size = config['new_size']

    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_dir = dataset_path
        image_list = os.listdir(input_dir)
        for i, img_name in enumerate(image_list):
            if i < len(image_list):
                img = os.path.join(input_dir, img_name)
                # with open(os.path.join(input_dir, img_name.replace('png', 'json')), 'r') as f:
                #     image_info = json.load(f)
                #     timestamp = image_info["cam_tstamp"]
                #     img_name_by_json = str(timestamp) + '.png'
                if 'png' in img or 'jpg' in img:
                    image = Image.open(img).convert('RGB')
                    im_size = image.size
                    if im_size == (1920, 1208):
                        image = image.crop((0, 248, im_size[0], im_size[1]))
                    image = Variable(transform(image).unsqueeze(0).cuda())

                    content, _ = encode(image)
                    outputs = decode(content)
                    outputs = (outputs + 1) / 2.
                    # path = os.path.join(output_path, 'x_n2', img_name_by_json)
                    # path = os.path.join(output_path, 'x_n2', img_name)
                    path = os.path.join(output_path, img_name)

                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
                    # vutils.save_image(image.data, os.path.join(output_path, 'x_n1', img_name_by_json),
                    #                   padding=0, normalize=True)
                    # print(img_name_by_json)    

def day2night(dataset_path, output_path):
    # parser = argparse.ArgumentParser()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    config_file = dir_path + '/configs/unit_day2night_test.yaml'
    config = get_config(config_file)
    trainer = UNIT_Trainer(config)
    trainer.cuda()
    trainer.eval()

    # state_dict = torch.load(dir_path + '/models/gen.pt')
    state_dict = torch.load('./models/generator_models/UNIT/gen.pt')

    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])


    encode = trainer.gen_a.encode
    style_encode = trainer.gen_b.encode
    decode = trainer.gen_b.decode
    new_size = config['new_size']

    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_dir = dataset_path
        image_list = os.listdir(input_dir)
        for i, img_name in enumerate(image_list):
            if i < len(image_list):
                img = os.path.join(input_dir, img_name)
                with open(os.path.join(input_dir, img_name.replace('png', 'json')), 'r') as f:
                    image_info = json.load(f)
                    timestamp = image_info["cam_tstamp"]
                    img_name_by_json = str(timestamp) + '.png'
                if 'png' in img or 'jpg' in img:
                    image = Image.open(img).convert('RGB')
                    im_size = image.size
                    if im_size == (1920, 1208):
                        image = image.crop((0, 248, im_size[0], im_size[1]))
                    image = Variable(transform(image).unsqueeze(0).cuda())

                    content, _ = encode(image)
                    outputs = decode(content)
                    outputs = (outputs + 1) / 2.
                    path = os.path.join(output_path, 'x_n2', img_name_by_json)
                    # path = os.path.join(output_path, 'x_n2', img_name)
                    # path = os.path.join(output_path, img_name)

                    vutils.save_image(outputs.data, path, padding=0, normalize=True)
                    vutils.save_image(image.data, os.path.join(output_path, 'x_n1', img_name_by_json),
                                      padding=0, normalize=True)
                    # print(img_name_by_json)

def image_control(object, transformation, parameter, dataset_path, semantic_path, output_path, **kwargs):
    if  'night' in parameter:
        day2night(dataset_path, output_path)