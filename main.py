import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.autograd import Variable
from torchvision import models, transforms
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import cPickle
from glob import glob
import os
import sys
import math
from pprint import pprint

import huva
from huva import plate_localizer, clip, rfcn, LazyImage, make_multiple
from huva.np_util import *
from huva.th_util import th_get_jet, get_model_param_norm, MonitoredAdam

from plate_gen.main import get_sample
from plate_gen.plate import char_to_charid, charid_to_char, gen_img_from_seq

from data import db2
from data import traffic_mixins

db = None

def load_db():
    global db
    db = db2.Database.load('db2.pkl')

def overnight_train():
    make('model8')
    train()
    set_learning_rate(0.0001)
    num_epoch = 10
    train()
    torch.save(model, 'model8.pth')

file_folder = os.path.dirname(os.path.realpath(__file__))

batch_size = 64
num_epoch = 30
input_W = 192
input_H = input_W / 3
downsample_times = 1
label_W = input_W / downsample_times
label_H = input_H / downsample_times
num_foreground = 37
num_classes = num_foreground + 1 # +1 background

black_white = True   # whether we are running a black&white model instead of RGB
num_input_cn = 1 if black_white else 3
model_name = None    # name of the model, used for archiving and evaluationg
model = None         # actual model
criterion = None
optimizer = None

heat_strategy = 'center' # center or full
heat_radius = 2
"""
full: 0.0008 is as low as it gets
center: 0.0002 quite low
"""

clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
def apply_clahe(img):
    if random.randint(0,1):
        img = clahe1.apply(img)
    else:
        img = clahe2.apply(img)
    return img

def change_contrast(img, base, target_range):
    valmax = img.max()
    valmin = img.min()
    valrange = valmax - valmin
    img = (img - valmin) / valrange * target_range + base
    return img

def mess_contrast(img):
    target_range = random.randint(40,255)
    base = random.randint(0, 255 - target_range)
    return change_contrast(img, base, target_range)


def get_synthetic_sample():
    """
    1. get img and coords from get_sample, cast to black-white if needed
    2. resize
    3. Generate label
    """
    """ load """
    global black_white
    img_np, coords = get_sample()
    H, W = img_np.shape[:2]
    # note that get_sample is in RGB format!!
    if black_white:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = img_np.reshape(H,W,1) # (H,W) -> (H,W,1)
    """ We do not crop """
    """ Resize to input_H by input_W """
    factor_Y = input_H / float(H)
    factor_X = input_W / float(W)
    img_resized = cv2.resize(img_np, (input_W, input_H))
    if black_white:
        img_resized = img_resized.reshape(input_H, input_W, 1)
    if random.randint(0,1):
        img_resized = np.expand_dims(apply_clahe(img_resized), 2)
    img   = torch.from_numpy(img_resized.transpose([2,0,1])).float()
    # some augmentation
    if random.randint(0,1): # invert bw
        img = 255 - img
    if random.randint(0,1): # mess with contrast
        img = mess_contrast(img)
    """ Generate heat label """
    label_np = np.zeros((num_classes, label_H, label_W), np.float32)
    for char in coords:
        cls_idx = char['class']
        assert cls_idx != 10
        corners = char['corners']
        pts = np.asarray(char['corners']).astype(np.int32)
        pts[:,0] = pts[:,0] * factor_X / downsample_times
        pts[:,1] = pts[:,1] * factor_Y / downsample_times
        if heat_strategy == 'full':
            cv2.fillPoly(label_np[cls_idx], [pts], 1.0)
        elif heat_strategy == 'center':
            centerx = int(round(pts[:,0].mean()))
            centery = int(round(pts[:,1].mean()))
            ss = heat_radius
            label_np[cls_idx][centery-ss:centery+ss+1, centerx-ss:centerx+ss+1] = 1
        else: 
            assert False, 'unknown heating strategy'
    # generate the background class
    label_np[num_classes-1] = 1 - label_np.max(0)
    label = torch.from_numpy(label_np)
    assert img.type() == 'torch.FloatTensor'
    assert label.type() == 'torch.FloatTensor'
    assert label.max() == 1.0
    return img, label

def output_pair(img_th, label_th, folder, img_idx):
    """
    img: torch Tensor, 1xH1xW1
    label: torch Tensor, nxH2xW2
    """
    img   = img_th.float().numpy().astype(np.uint8)
    label = label_th.float().numpy()
    cv2.imwrite('{}/{}.jpg'.format(folder, img_idx), img[0])
    H,W = img[0].shape[:2]
    # cid for charid
    for cid in xrange(num_classes):
        label_cid = label[cid]
        if label_cid.max() > 0.2:
            char = '' if cid==num_classes-1 else charid_to_char(cid)
            jet_cid = huva.np_util.get_jet(img[0].reshape(H,W,1), cv2.resize(label_cid, (W,H)))
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, img_idx, char), jet_cid)
    #label_max = label.max(0)
    #jet_max = huva.np_util.get_jet(img[0].reshape(H,W,1), cv2.resize(label_max, (W,H)))
    #cv2.imwrite('{}/{}_{}.jpg'.format(folder, img_idx, '_Zax'), jet_max)





"""
*************************************** Dataset ************************************************
"""

class PlateGenDataset():
    def __init__(self, length):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        if i % 3 == 0:
            return get_real_labelled_sample()
        else:
            return get_synthetic_sample()

# do mean subtraction, BGR order
mean_bgr = torch.FloatTensor([104, 117, 124]).view(1,3,1,1)
mean_bw  = torch.FloatTensor([124]).view(1,1,1,1)

if __name__=='__main__':
    dataset  = PlateGenDataset(10000) # try 10K epoch size
    loader   = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)

"""
*************************************** Model ************************************************
"""

execfile(os.path.join(file_folder, 'model.py'))

"""
*************************************** Training ************************************************
"""

def make(name):
    global model_name, model, criterion, optimizer
    if os.path.exists('{}.pth'.format(name)):
        print('Warning: {} already exits'.format(name))
    model_name = name
    model     = VGGLikeHeatRegressor(32).cuda()
    make_others()

def make_others():
    global criterion, optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = MonitoredAdam(model.parameters(), 0.001, weight_decay=0.00001)

def load_model(_model_name):
    global model_name, model, criterion, optimizer
    model = torch.load('{}.pth'.format(_model_name))
    model_name = _model_name
    make_others()

int_report = 1
int_output = 600
def train():
    total_batch_count = 0
    min_loss = 99999
    min_loss_batches = 0
    model.train()
    for epoch in xrange(num_epoch):
        g_loss = 0.0
        for batch, (imgs, labels) in enumerate(loader):
            will_report = batch % int_report == 0
            #
            v_imgs = Variable(imgs - (mean_bw if black_white else mean_bgr).expand_as(imgs)).cuda()
            v_labels = Variable(labels).cuda()
            v_output = model(v_imgs)
            v_loss   = criterion(v_output, v_labels)
            #
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step(monitor_update=will_report)
            g_loss += v_loss.data[0]
            if will_report:
                avg_loss = g_loss / int_report
                if avg_loss < min_loss:
                    min_loss = avg_loss
                    min_loss_batches = 0
                else:
                    min_loss_batches += 1
                print('{}, {}, {:6f} [{:6f}/{:6f}] [{}]'.format(
                    epoch, batch, avg_loss, get_model_param_norm(model), optimizer.update_norm, min_loss_batches))
                g_loss = 0.0
            total_batch_count += 1
            if total_batch_count % int_output == 0:
                output = v_output.data.cpu()
                os.system('rm ./imgs/*')
                for i in range(batch_size):
                    output_pair(imgs[i], v_output.data[i].cpu(), 'imgs', i)
                evaluate_on_folder()

def set_learning_rate(lr):
    """
    Since we only have one group, it's enough to do optimizer.param_groups[0]['lr'] = lr
    but do the proper thing anyway
    """
    for group in optimizer.param_groups:
        group['lr'] = lr

"""
*************************************** Testing ************************************************
"""

def scale_img(img, W_target=160):
    """
    Scale an image such that its width is close to W_target, and that both sides are multiples of 24
    """
    H,W = img.shape[:2]
    W_new = make_multiple(W_target, 24)
    H_proposed = H * ( W_new /float(W) )
    H_new = make_multiple(H_proposed, 24)
    img_new = cv2.resize(img, (W_new, H_new))
    if img_new.ndim != 3:
        img_new = np.expand_dims(img_new, 2)
    return img_new

def get_scaled_crop(plate_crop):
    """
    Find a reasonable size for a plate_crop
    """
    assert plate_crop.ndim == 2
    H,W = plate_crop.shape[:2]
    if W/float(H) > 2: # 1-line
        plate_crop = scale_img(plate_crop, 140*1.6)
    else: # 2-line
        plate_crop = scale_img(plate_crop, 110*1.6)
    return plate_crop


plate_crops_folder = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/plate_crops_3000'

def pass_imgs(imgs_np):
    """
    Pass the images through currently loaded model, and return the heatmaps
    imgs is numpy array of NxHxWxnum_channels
    """
    #model.eval()
    imgs = torch.from_numpy(imgs_np.transpose([0,3,1,2])).float()
    v_imgs = Variable(imgs - (mean_bw if black_white else mean_bgr).expand_as(imgs)).cuda()
    v_outs = model(v_imgs)
    return v_outs.data.cpu()


execfile(os.path.join(file_folder, 'evaluate.py'))


def get_good_crop_region(car_bbox, plate_bbox, img):
    """
    Takes a reasonably sized crop of the plate
    - 1-line plate: horozontally expand 0.3 randomly, vertically 1/3 of that
    - 2-line plate: vertically expand 0.3 randomly, horizontally 3x of that
    """
    assert isinstance(car_bbox, db2.BBox)
    assert isinstance(plate_bbox, db2.BBox)
    # cx,cy,cw,ch = car_bbox.xywh()
    px,py,pw,ph = plate_bbox.xywh()
    centerx = px + pw/2
    centery = py + ph/2
    # determine crop size
    H_target = make_multiple(ph * 1.5, 24)
    W_target = make_multiple(pw * 1.5, 24)
    """
    is_2line = (pw / float(ph)) < 2.4
    if is_2line: # 2 line
        H_target = make_multiple(int(ph * 1.3), 24)
        W_target = make_multiple(3 * H_target, 24)
        assert H_target >= ph
        assert W_target >= pw
    else:        # one line
        W_target = int(pw * 1.3)
        H_target = int(W_target / 3)
        assert H_target >= ph
        assert W_target >= pw
    """
    # choose a center crop
    x1 = centerx - W_target / 2
    x2 = centerx + W_target / 2
    y1 = centery - H_target / 2
    y2 = centery + H_target / 2
    x1 = clip(x1, 0, img.shape[1])
    x2 = clip(x2, 0, img.shape[1])
    y1 = clip(y1, 0, img.shape[0])
    y2 = clip(y2, 0, img.shape[0])
    return x1,y1,x2,y2

def place_dot(frame_img, plate_crop, (x1,y1,x2,y2), blob, just_recover=False):
    x = int(float(blob.centerx) / plate_crop.shape[1] * (x2-x1) + x1)
    y = int(float(blob.centery) / plate_crop.shape[0] * (y2-y1) + y1)
    if just_recover:
        return x,y
    else:
        cv2.circle(frame_img, (x, y), 3, (255,0,0), 1)

def merge_plate_img_with_label(scaled_crop, seq):
    """
    Slap the sequence under the plate crop to get a new image
    """
    charid_sequence = map(char_to_charid, seq.get_str_seq())
    seq_img = gen_img_from_seq(charid_sequence, 0)[0][:,:,0] # using font0=UKFont
    H = scaled_crop.shape[0] + seq_img.shape[0]
    W = max(scaled_crop.shape[1], seq_img.shape[1])
    output_img = np.zeros((H,W), np.uint8)
    output_img[:scaled_crop.shape[0], :scaled_crop.shape[1]] = scaled_crop[:,:,0]
    output_img[scaled_crop.shape[0]:, :seq_img.shape[1]] = seq_img
    return output_img

def auto_label_plates(min_area=3000, max_area=99999, num_skip=0, show=True, folder_name=None, warn_if_in={}):
    """
    Find every plate_bbox whose area is greater than min_area,
    Feed it through the model to get the sequence
    num_skip: number of frames to skip. For fast-forwarding
    folder_name: if specified, only operate on this folder
    warn_if_in: a dictionary of string_seq -> True. If detected plate is in this set, produce a warning
    """
    output_folder = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/manual_batches/plate_auto_2000to3000'
    skip_count = 0
    folder_keys = sorted(db.folder_registry.keys())
    for folder_key in folder_keys:
        folder = db.folder_registry[folder_key]
        print(folder_key)
        if folder_name != None and folder_name not in folder_key: continue
        for frame in folder.frames:
            if not frame.has_run_rfcn(): continue
            lazy_img = LazyImage(frame.absolute_path())
            if skip_count < num_skip:
                skip_count += 1
                continue
            for car_bbox in frame.parts:
                if car_bbox.name != 'car': continue
                for plate_bbox in car_bbox.parts:
                    if plate_bbox.name != 'plate': continue
                    if plate_bbox.label_type() not in ['manual', 'auto']: continue
                    _,_,w,h = plate_bbox.xywh()
                    if w*h < min_area or w*h > max_area: continue
                    """ crop region """
                    x1,y1,x2,y2 = get_good_crop_region(car_bbox, plate_bbox, lazy_img.get())
                    plate_crop = cv2.cvtColor(lazy_img.get()[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    scaled_crop = get_scaled_crop(plate_crop)
                    """ pass model """
                    heatmaps = pass_imgs(np.expand_dims(scaled_crop, 0)).numpy()
                    sequences, filtered = infer_sequences(heatmaps[0])
                    frame_img = lazy_img.get().copy()
                    if len(filtered)==0: 
                        if len(sequences) == 0:
                            print('Completely missed')
                        else:
                            print('Missed, top answer: {}'.format(sequences[0].get_str_seq()))
                    else:
                        seq = filtered[0]
                        string_seq = seq.get_str_seq()
                        print(string_seq)
                        if string_seq in warn_if_in:
                            print('Warning: {} exists in the warning set'.format(string_seq))
                        auto_sequence = []
                        for blob in seq.blobs:
                            x,y = place_dot(None, scaled_crop, (x1,y1,x2,y2), blob, just_recover=True)
                            auto_sequence.append((x,y, blob.char))
                        plate_bbox.auto_sequence(auto_sequence)
                        """ write it out """
                        output_img = merge_plate_img_with_label(scaled_crop, seq)
                        cv2.imwrite(os.path.join(output_folder, plate_bbox.unique_name(with_jpg=True)), output_img)
                    if show:
                        cv2.rectangle(frame_img, (x1,y1), (x2,y2), (255,0,0), 3)
                        fix, ax = plt.subplots(figsize=(20,11))
                        ax.imshow(frame_img, aspect='equal')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()

def auto_load_plates():
    input_folder = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/manual_batches/plate_auto_2000to3000'
    filenames = glob(input_folder + '/*.jpg')
    for filename in filenames:
        name = filename.split('/')[-1].split('.')[0]
        parts = map(int, name.split('#'))
        folder_id, frame_id, car_id, plate_id = parts
        folder = db.get_folder_by_id(folder_id)
        frame = folder.frames[frame_id]
        car_bbox = frame.parts[car_id]
        plate_bbox = car_bbox.parts[plate_id]
        assert plate_bbox.has_auto_sequence()
        plate_bbox.auto_sequence_confirmed(True)
        print(parts)
    print(len(filenames))

def extract_real_labelled_plates():
    output_folder = real_labelled_plates_folder
    name_to_sequence = {}
    for folder in db.get_folders():
        for frame in folder.frames:
            if not frame.has_run_rfcn(): continue
            lazy_img = LazyImage(frame.absolute_path())
            for car_bbox in frame.parts:
                if car_bbox.name != 'car': continue
                for plate_bbox in car_bbox.parts:
                    if plate_bbox.name != 'plate': continue
                    if not plate_bbox.auto_sequence_confirmed(): continue
                    auto_sequence = plate_bbox.auto_sequence()
                    """ compute crop boundary """
                    img = lazy_img.get()
                    x1 = clip(min([x for x,y,char in auto_sequence]) - 100, 0, img.shape[1])
                    y1 = clip(min([y for x,y,char in auto_sequence]) - 100, 0, img.shape[0])
                    x2 = clip(max([x for x,y,char in auto_sequence]) + 100, 0, img.shape[1])
                    y2 = clip(max([y for x,y,char in auto_sequence]) + 100, 0, img.shape[0])
                    new_sequence = [(x-x1,y-y1, char) for x,y,char in auto_sequence]
                    img_crop = img[y1:y2, x1:x2]
                    imname = plate_bbox.unique_name(with_jpg=True)
                    cv2.imwrite(os.path.join(output_folder, imname), img_crop)
                    name_to_sequence[imname] = new_sequence
    cPickle.dump(name_to_sequence, open(os.path.join(output_folder, 'name_to_sequence.pkl'), 'wb'))

real_labelled_plates_folder = '/home/noid/data/huva_real_labelled_plates'
name_to_sequence = cPickle.load(open(os.path.join(real_labelled_plates_folder, 'name_to_sequence.pkl')))

def get_real_labelled_sample(show=False):
    length = len(name_to_sequence)
    idx = random.randint(0, length-1)
    imname = name_to_sequence.keys()[idx]
    sequence = name_to_sequence[imname]
    img = cv2.imread(os.path.join(real_labelled_plates_folder, imname), 0)
    """ (xmin,ymin) (xmax,ymax) is the minimum region that must be included in the crop """
    xmin = clip(min([x for x,y,char in sequence]) - 10, 0, img.shape[1])
    ymin = clip(min([y for x,y,char in sequence]) - 10, 0, img.shape[0])
    xmax = clip(max([x for x,y,char in sequence]) + 10, 0, img.shape[1])
    ymax = clip(max([y for x,y,char in sequence]) + 10, 0, img.shape[0])
    """
    compute crop region
       left-extend random amount, right-extend random amount, clip
       divide width by 3 to get height
    """
    x1 = clip(xmin - random.randint(5,int((xmax-xmin)*0.5)), 0, img.shape[1])
    x2 = clip(xmax + random.randint(5,int((xmax-xmin)*0.5)), 0, img.shape[1])
    width = x2 - x1
    height = int(round(width / 3.0))
    """ if this is a 2-line plate, it's possible that height < ymax-ymin """
    if height < ymax - ymin:
        y1 = clip(ymin - random.randint(5, int((ymax-ymin)*0.5)), 0, img.shape[0])
        y2 = clip(ymax + random.randint(5, int((ymax-ymin)*0.5)), 0, img.shape[0])
        height = y2-y1
        width = int(round(height * 3.0))
        while width < xmax - xmin:
            print(width, xmax, xmin, img.shape)
            width = xmax - xmin
        x1 = xmin - random.randint(0, width - (xmax - xmin))
        x2 = x1 + width
    else:
        y1 = ymin - random.randint(0, height - (ymax - ymin))
        y2 = y1 + height
    x1 = clip(x1, 0, img.shape[1])
    y1 = clip(y1, 0, img.shape[0])
    x2 = clip(x2, 0, img.shape[1])
    y2 = clip(y2, 0, img.shape[0])
    img_crop = img[y1:y2, x1:x2]
    height, width = img_crop.shape[:2]
    """ resize image """
    factor_x = input_W / float(width)
    factor_y = input_H / float(height)
    new_sequence = [(int((x-x1)*factor_x),int((y-y1)*factor_y),char) for x,y,char in sequence]
    img_resized = cv2.resize(img_crop, (input_W, input_H))
    """ heat it """
    label_np = np.zeros((num_classes, label_H, label_W), np.float32)
    ss = heat_radius
    for x,y,char in new_sequence:
        charid = char_to_charid(char)
        label_np[charid, y-ss:y+ss+1, x-ss:x+ss+1] = 1.0
    label_np[num_classes-1] = 1 - label_np.max(0)
    """ visualize if asked """
    if show:
        for x,y,char in new_sequence:
            img_resized[y-ss:y+ss+1, x-ss:x+ss+1] = 255
        plt.imshow(img_resized)
        plt.show()
    label = torch.from_numpy(label_np)
    """ some augmentation """
    if random.randint(0,1):
        img_resized = apply_clahe(img_resized)
    img = torch.from_numpy(np.expand_dims(img_resized, 0)).float()
    if random.randint(0,1):
        img = mess_contrast(img)
    return img, label


"""
======================================== For integrated use ====================================================
"""

demo_root = '/mnt/hdd3t/data/huva_cars/20170206-lornie-road/demos/number_sequence'

def sequence_demo(show=False, show_small=False, write_out=False, draw_mispredict=False, mode='detect'):
    """
    1. For every image in output_root/with_plates/*.jpg
    2. Get all the plate proposal regions from name_to_boxes.pkl
    3. For each proposed region, try to find a license sequence
    4. If found, draw it onto the image
    5. Output the annotated image to output_root/with_numbers_2/*.jpg
    6. Also count number of proposals and numbers of plates detected
    -- model6: 17466 / 10302 (margin=0.15, base=60, target_range=135)
       - (0.35, 0.15), (110)
    -- model8: 17466 / 12179 (margin=0.15, base=60, target_range=135)
       - (0.35, 0.15), (110)
    """
    output_root = demo_root
    raw_filenames = sorted(glob(output_root+'/raw/*.jpg'))
    name_to_boxes = cPickle.load(open(os.path.join(output_root, 'with_plates', 'name_to_boxes.pkl')))
    short_names = sorted(name_to_boxes.keys())
    """ Count (number of plate proposals, number of plates detected) """
    if mode=='detect':
        num_plates_proposed = 0
        num_plates_detected = 0
    elif mode=='integrate':
        fidx_to_car_insts = []
        integrator = LPRFrameIntegrator()
    for fidx, short_name in enumerate(short_names):
        img_color = cv2.imread(os.path.join(output_root, 'raw', short_name))
        img = cv2.imread(os.path.join(output_root, 'raw', short_name), 0)
        boxes = name_to_boxes[short_name]
        if mode=='integrate':
            car_insts = [] # [(x1,y1,x2,y2), car_info] for LPRFrameIntegrator
        for box in boxes:
            x1,y1,x2,y2 = box
            w = x2 - x1
            h = y2 - y1
            """ expand the crop box """
            w_ratio = 0.15
            h_ratio = w_ratio
            if w/h < 2:
                w_ratio = 0.35
                h_ratio = 0.15
            x1 = clip(int(x1 - w*w_ratio), 0, img.shape[1])
            x2 = clip(int(x2 + w*w_ratio), 0, img.shape[1])
            y1 = clip(int(y1 - h*h_ratio), 0, img.shape[0])
            y2 = clip(int(y2 + h*h_ratio), 0, img.shape[0])
            w = x2 - x1
            h = y2 - y1
            if w<10 or h < 10: continue
            """ crop """
            plate_crop = img[y1:y2, x1:x2]
            if show_small:
                print('plate_crop')
                plt.imshow(plate_crop, cmap='gray');
                plt.show()
            """ resize crop """
            plate_crop_scaled = get_scaled_crop(plate_crop)
            if show_small:
                print('plate_crop_scaled')
                plt.imshow(plate_crop_scaled[:,:,0], cmap='gray');
                plt.show()
            """ enhance crop """
            plate_crop_scaled = plate_crop_scaled.astype(np.float32)
            plate_crop_scaled = change_contrast(plate_crop_scaled, 60, 135)
            """ pass to model and get sequence """
            heatmaps = pass_imgs(np.expand_dims(plate_crop_scaled, 0)).numpy()
            sequences, filtered = infer_sequences(heatmaps[0])
            """ mode-specific stuffs """
            if mode=='detect':
                num_plates_proposed += 1
                if write_out:
                    cv2.rectangle(img_color, (x1,y1), (x2,y2), (0,0,255), 3)
                if len(filtered) != 0:
                    num_plates_detected += 1
                    str_seq = filtered[0].get_str_seq()
                    if write_out:
                        cv2.putText(img_color, str_seq, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                elif write_out and draw_mispredict and len(sequences)>0:
                    str_seq = sequences[0].get_str_seq()
                    cv2.putText(img_color, str_seq, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            elif mode=='integrate':
                x = (x1+x2)/2
                y = (y1+y2)/2
                # find car_info of previous car
                if fidx==0: # just create new car
                    car_info = CarInfo()
                else:
                    prev_car_instance = find_nearest_car(box, fidx_to_car_insts[fidx-1])
                    if prev_car_instance is None:
                        car_info = CarInfo()
                    else:
                        car_info = prev_car_instance[1]
                if len(filtered):
                    car_info.add_plate(box, filtered[0].get_str_seq(), filtered[0].prob)
                car_instance = (box, car_info)
                car_insts.append(car_instance)
        print(short_name)
        if mode=='detect' and write_out:
            out_path = os.path.join(output_root, 'with_numbers_2', short_name)
            cv2.imwrite(out_path, img_color)
        elif mode=='integrate':
            fidx_to_car_insts.append(car_insts)
            integrator.add_frame(short_name, img_color, car_insts)
        else:
            assert False, 'unknown mode={}'.format(mode)
    if mode=='detect':
        print(num_plates_proposed, num_plates_detected)
    elif mode=='integrate':
        integrator.flush()

def find_nearest_car(box, prev_car_insts, threshold=100):
    x1,y1,x2,y2 = box
    x = (x1+x2)/2
    y = (y1+y2)/2
    nearest_dist = 9999
    nearest_car  = None
    for pbox, pcar_info in prev_car_insts:
        px1,py1,px2,py2 = pbox
        px = (px1 + px2) / 2
        py = (py1 + py2) / 2
        dist = math.sqrt( (x-px)**2 + (y-py)**2 )
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_car  = (pbox, pcar_info)
    if nearest_dist < threshold:
        return nearest_car
    else:
        return None

class CarInfo:
    def __init__(self):
        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        while sum(self.color) < 150:
            self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.plates = [] # [(x1,y1,x2,y2, str_seq, prob)]
        self.likely_sequence = None
    def add_plate(self, (x1,y1,x2,y2), str_seq, prob):
        assert self.likely_sequence is None, 'cannot add plate after get_likely_sequence has been called'
        self.plates.append(((x1,y1,x2,y2), str_seq, prob))
    def get_likely_sequence(self):
        if self.likely_sequence is None:
            strseq_to_score = {}
            for plate in self.plates:
                (x1,y1,x2,y2), str_seq, prob = plate
                y = (y1+y2)/2
                score = y # TODO prob * y
                if str_seq not in strseq_to_score:
                    strseq_to_score[str_seq] = 0
                strseq_to_score[str_seq] += score
            items = strseq_to_score.items()
            if len(items)==0:
                self.likely_sequence = ''
            else:
                likely_plate = sorted(items, reverse=True, key=lambda (strseq, score):score)[0]
                if likely_plate[0] in ['AH662K', 'SJL799Y', 'S3279T']:
                    print(self.plates)
                self.likely_sequence = likely_plate[0]
        return self.likely_sequence

from collections import deque
class LPRFrameIntegrator:
    def __init__(self):
        self.frames = deque(maxlen=100)
    def add_frame(self, short_name, frame_img, car_insts):
        """
        car_insts: [((x1,y1,x2,y2), CarInfo)]
        CarInfo: (color, [PlateInfo])
        PlateInfo: (str_seq, (x,y), prob)
        """
        self.frames.appendleft((short_name, frame_img, car_insts))
        while len(self.frames) > 90:
            self.pop_frame()
    def pop_frame(self):
        short_name, frame_img, car_insts = self.frames.pop()
        """ draw the cars """
        for (x1,y1,x2,y2), car_info in car_insts:
            color = car_info.color
            strseq= car_info.get_likely_sequence()
            x1 -= int(0.2 * (x2-x1))
            x2 += int(0.2 * (x2-x1))
            y1 -= int(0.2 * (y2-y1))
            y2 += int(0.2 * (y2-y1))
            cv2.rectangle(frame_img, (x1,y1), (x2,y2), color, 3)
            (w,h),baseline = cv2.getTextSize(strseq, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
            cv2.rectangle(frame_img, (x1, y1-5-h), (x1+w, y1-3), (0,0,0), thickness=cv2.cv.CV_FILLED)
            cv2.putText(frame_img, strseq, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        output_root = demo_root
        outpath = os.path.join(output_root, 'with_plate_tracking', short_name)
        cv2.imwrite(outpath, frame_img)
    def flush(self):
        while len(self.frames):
            self.pop_frame()



def test():
    for img_idx in range(64):
        if img_idx < 32:
            img_th, label_th = get_synthetic_sample()
        else:
            img_th, label_th = get_real_labelled_sample()
        output_pair(img_th, label_th, 'imgs', img_idx)

if __name__ == '__main__':
    test()
