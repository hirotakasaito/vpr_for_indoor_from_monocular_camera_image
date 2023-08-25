import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import animation
import torchvision
import os

trans = torchvision.transforms.ToPILImage()

def get_concat_v(im1, im2, im3):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height+im2.height))

    return dst

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_h_multi(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h(_im, im)
    return _im

def visualize_img(img1, img2, img3):

    img1 = img1.permute(0,3,1,2).to('cpu').detach()
    img2 = img2.permute(0,3,1,2).to('cpu').detach()
    img3 = img3.permute(0,3,1,2).to('cpu').detach()

    imgs1 = []
    imgs2 = []
    imgs3 = []

    for i1, i2, i3 in zip(img1, img2, img3):
        imgs1.append(trans(i1[[2,1,0],:,:]))
        imgs2.append(trans(i2[[2,1,0],:,:]))
        imgs3.append(trans(i3[[2,1,0],:,:]))

    imgs1 = get_concat_h_multi(imgs1)
    imgs2 = get_concat_h_multi(imgs2)
    imgs3 = get_concat_h_multi(imgs3)

    output_img = get_concat_v(imgs1, imgs2, imgs3)
    return output_img

def save_figure(out_list, img_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (o, img)  in enumerate(zip(out_list, img_list)):

        fig = plt.figure(figsize=(20,10))
        fig.subplots_adjust(bottom=0.15)
        # text = f'anchor: {o[0]:.2f},positive: {o[1]:.2f},negative: {o[2]:.2f}' 
        text = f'anchor-positive: {o[0]:.2f},anchor-negative: {o[1]:.2f}' 
        # fig.text(0.45, 0.02, text, fontsize=12)
        
        #current img
        ax_c_img = fig.add_subplot(1,1,1)
        # ax_c_img.imshow(img[0][:,:,[2,1,0]])
        ax_c_img.imshow(img)
        ax_c_img.set_xlabel(text, fontsize=15)
        ax_c_img.axes.xaxis.set_ticks([])
        ax_c_img.axes.yaxis.set_ticks([])

        ##goal img
        #ax_g_img = fig.add_subplot(1,2,2)
        #ax_g_img.imshow(img[1][:,:,[2,1,0]])
        #ax_g_img.set_xlabel('image2', fontsize=12)
        #ax_g_img.axes.xaxis.set_ticks([])
        #ax_g_img.axes.yaxis.set_ticks([])

        fig.savefig(os.path.join(save_dir, str(i) + ".png"), bbox_inches='tight')
        fig = plt.figure()

