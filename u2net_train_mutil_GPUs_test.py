



import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import logging
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                   format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                   # filename = './log/logging.log',
                   filemode = 'a',
                   )

# 将单卡单GPU改成单卡多GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))
	logging.info("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = os.path.join('/nfs/volume-821-5/huxiaoliang/synthesis_segment_datase_no_bg_blur' + os.sep)
# print('*'*80)
# print(os.path.exists(data_dir))
tra_image_dir = os.path.join('img' + os.sep)
tra_label_dir = os.path.join('img_mask' + os.sep)
# tra_image_dir = os.path.join('test_img' + os.sep)
# tra_label_dir = os.path.join('test_img_mask' + os.sep)
# data_dir = os.path.join('/nfs/private/datasets/DUTS-TR' + os.sep)
# tra_image_dir = os.path.join('DUTS-TR-Image' + os.sep)
# tra_label_dir = os.path.join('DUTS-TR-Mask' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

# 直接训练
model_dir = os.path.join('/nfs/volume-821-5/huxiaoliang/modelfiles/u2net-saved_models', model_name + os.sep)
model_path = os.path.join(model_dir,model_name + '.pth')

# 调参数
epoch_num = 100000 # 100000->10000
batch_size_train = 80
batch_size_val = 20
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

logging.info("---")
logging.info("train images: {}".format(len(tra_img_name_list)))
logging.info("train labels: {}".format(len(tra_lbl_name_list)))
logging.info("---")



train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

USE_CUDA = torch.cuda.is_available()
print('='*80)
print(USE_CUDA)
device = torch.device("cuda:0" if USE_CUDA else "cpu")


from collections import OrderedDict
def load_pretrainedmodel(model_path, model_):
    pre_model = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
    #print(pre_model)
    if USE_CUDA:
        state_dict = OrderedDict()
        for k in pre_model.state_dict():
            name = k
            if name[:7] != 'module' and torch.cuda.device_count() > 1: # loaded model is single GPU but we will train it in multiple GPUS!
                name = 'module.' + name #add 'module'
            elif name[:7] == 'module' and torch.cuda.device_count() == 1: # loaded model is multiple GPUs but we will train it in single GPU!
                name = k[7:]# remove `module.`
            state_dict[name] = pre_model.state_dict()[k]
            #print(name)
        model_.load_state_dict(state_dict)
        #model_.load_state_dict(torch.load(modelname)['model'].state_dict())
    else:
        model_ = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
    return model_
if USE_CUDA:
    net = nn.DataParallel(net,device_ids=[0,1,2,3])
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    # net = load_pretrainedmodel(model_path, net)
    # logging.info('---GPUs training---')
    # net.cuda()
    # pytorch禁用cudnn
    # torch.backends.cudnn.benchmark = False

# ------- 4. define optimizer --------
logging.info("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
logging.info("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations  2000->200
log_frq = 10 # logging every 10 batches

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.data[0]
        running_loss += loss.item()
        # running_tar_loss += loss2.data[0]
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        # if ((i + 1) * batch_size_train) % log_frq:
            # print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            # epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        logging.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f ]" % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))


        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0