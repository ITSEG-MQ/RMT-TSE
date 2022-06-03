import matplotlib
# matplotlib.use('Agg')
from model_a2d2 import Epoch, weight_init, build_vgg16, build_resnet101, Vgg16, Resnet101
from data_a2d2 import A2D2Dataset,A2D23D, A2D2Diff, A2D2MT
import torch.optim as optim
import torch.nn as nn
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import csv
from os import path
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
from torch.optim import lr_scheduler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def auto_test():
    mt_paths = ["bicycle_50"]
    models = ["epoch","vgg16","resnet101"]
    for model in models:
        for mt in mt_paths:
            print(mt,model)
            test_mt(mt, model)
 
def test_mt(mt_path,model_name="vgg16"):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if model_name == "vgg16":
        model = Vgg16()
    elif model_name == "epoch":
        model = Epoch()
    elif model_name == "resnet101":
        model = Resnet101()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(torch.load(model_name+'-30.pt'))
    model.to(device)
    test_composed = transforms.Compose([transforms.ToTensor()])
    source_test_dataset = A2D2Diff(phase="test", transform=test_composed, root_path="..")
    # source_test_dataset = A2D2MT(root_path="..", source_path="car_100", mt_path="car_100", mode="self")
    follow_up_test_dataset = A2D2MT(root_path="..", source_path=mt_path, mt_path=mt_path, mode="self")
    # source_test_dataloader = DataLoader(source_test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # follow_up_test_dataloader = DataLoader(follow_up_test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        bg_speed = []
        label = []
        source_pred = []
        follow_up_pred = []
        for i in range(len(source_test_dataset)):
            source_images = source_test_dataset[i][0]
            source_bg_speed = source_test_dataset[i][1]
            source_label = source_test_dataset[i][2]

            follow_up_images = follow_up_test_dataset[i][0]
            follow_up_bg_speed = follow_up_test_dataset[i][1]
            follow_up_label = follow_up_test_dataset[i][2]
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
            # print(source_label, follow_up_label)
            # print(source_output, follow_up_output)
            # break
        # label = np.array(label).reshape(-1, 1)
        # source_pred = np.array(source_pred).reshape(-1, 1)
        # follow_up_pred = np.array(follow_up_pred).reshape(-1, 1)
        df = pd.DataFrame({"timestamp":follow_up_test_dataset.timestamp_list[:-1],"source_speed": bg_speed,"label": label, "source_pred": source_pred, "follow_up_pred": follow_up_pred})
        df.to_csv(model_name+"_"+mt_path + "_100_self.csv", index=False)

def test_model(model, device, test_generator):
    # test_loss = 0
    y_pred = []
    y_true = []
    # criterion = nn.L1Loss()
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for i, sample_batched in enumerate(test_generator):
            batch_image = sample_batched[0]
            batch_speed = sample_batched[1]
            batch_y = sample_batched[2]

            batch_image = batch_image.type(torch.FloatTensor)
            batch_speed = batch_speed.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_image = batch_image.to(device)
            batch_speed = batch_speed.to(device)

            batch_y = batch_y.to(device)

            outputs = model((batch_image, batch_speed)).view(-1)
            y_pred.append(outputs.item())
            y_true.append(batch_y.item())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        mse = np.mean((y_pred - y_true)**2)
        mae = np.mean(np.abs(y_pred - y_true))
        print("MAE loss:", mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training models")
    parser.add_argument('--model_name', action='store', type=str, required=True)
    parser.add_argument('--data_root', action='store', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--re_train', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    re_train = args.re_train
    test = args.test

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model_name = args.model_name
    if model_name == 'epoch':
        model = Epoch()
    elif model_name == 'vgg16':
        model = Vgg16()
    elif model_name == 'resnet101':
        model = Resnet101()

    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    if not test:
        train_composed = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_composed = transforms.Compose([transforms.ToTensor()])
        train_dataset = A2D2Diff(phase="train", transform=test_composed, root_path=args.data_root)
        validation_dataset = A2D2Diff(phase="validation", transform=test_composed, root_path=args.data_root)

        batch_size = args.batch_size
        epochs = args.epochs
        train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        validation_generator = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        criterion = nn.L1Loss()
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        optimizer = optim.Adam(params_to_update, lr=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        if re_train:
            model.load_state_dict(torch.load('models/driving_models/' + model_name + '.pt'))
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for step, sample_batched in enumerate(train_generator):
                batch_image = sample_batched[0]
                batch_speed = sample_batched[1]
                batch_y = sample_batched[2]
                batch_image = batch_image.type(torch.FloatTensor)
                batch_speed = batch_speed.type(torch.FloatTensor)
                batch_y = batch_y.type(torch.FloatTensor)
                batch_image = batch_image.to(device)
                batch_speed = batch_speed.to(device)

                batch_y = batch_y.to(device)
                outputs = model((batch_image, batch_speed)).view(-1)
                print("+", end="")
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = loss.item()
                total_loss += running_loss
                if step % 10 == 0:
                    print("Epoch %d Step %d MSE loss: %f" % (epoch, step, running_loss))
            exp_lr_scheduler.step()
                
            validation_loss = 0
            model.eval()
            with torch.no_grad():
                for i, sample_batched in enumerate(validation_generator):

                    batch_x = sample_batched[0]
                    batch_speed = sample_batched[1]
                    # print(batch_x.numpy())
                    batch_y = sample_batched[2]

                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_speed = batch_speed.type(torch.FloatTensor)

                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_x = batch_x.to(device)
                    batch_speed = batch_speed.to(device)

                    batch_y = batch_y.to(device)

                    outputs = model((batch_x, batch_speed)).view(-1)
                    print("-|", end="")
                    loss = criterion(outputs, batch_y)
                    running_loss = loss.item()
                    validation_loss += running_loss
            print('Epoch %d  training RMSE loss: %.4f test loss: %.4f' % (epoch,  total_loss / step, validation_loss / i))
            if ((epoch + 1) % 10 == 0):
                torch.save(model.state_dict(), 'models/driving_models/' + model_name + '.pt')
        
        # test_dataset = A2D2Diff(phase="test", transform=test_composed, root_path=args.dataroot)
        # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        test_model(model, device, validation_generator)
    else:
        model.load_state_dict(torch.load('models/driving_models/' + model_name + '.pt'))
        test_composed = transforms.Compose([transforms.ToTensor()])
        test_dataset = A2D2Diff(phase="test", transform=test_composed, root_path=args.dataroot)
        test_generator = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        test_model(model, device, test_generator)




