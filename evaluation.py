from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '3'

import cv2
import glob
import time
import caffe
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='evaluation tools')
    parser.add_argument('-proto', dest='proto', type=str, required=True, help='path to prototxt')
    parser.add_argument('-model', dest='model', type=str, required=True, help='path to caffemodel')
    parser.add_argument('-label', dest='label', type=str, required=True, help='path to label.txt')
    parser.add_argument('-imgs', dest='imgs', type=str, required=True, help='path to image root')
    parser.add_argument('-batch', dest='batch', type=int, required=True, help='eval batch size')
    parser.add_argument('-size', dest='size', type=int, required=True, help='net input size')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    img_size = args.size
    rsz_size = int(img_size / 0.875)
    diff = rsz_size - img_size
    diff_st = int(diff / 2)
    diff_ed = diff - diff_st

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    net.blobs['data'].reshape(args.batch, 3, img_size, img_size)
    
    gts = {}
    with open(args.label) as f:
        for line in f.readlines():
            path, label = line.split(' ')
            gts[path] = int(label)
    assert(len(gts) % args.batch == 0)

    img_list = glob.glob('%s/*JPEG' % (args.imgs[:-1] if args.imgs[-1] == '/' else args.imgs))

    start = time.time()
    net_time = 0
    count = 0
    iterations = len(gts) // args.batch
    for curr in range(iterations):
        st = curr * args.batch
        ed = (curr + 1) * args.batch
        img_slice = img_list[st: ed]

        labels = []
        for idx, img_path in enumerate(img_slice):
            img_name = os.path.split(img_path)[1]
            labels.append(gts[img_name])

            img = cv2.imread(img_path).astype(np.float32)

            for c in range(3):
                img[:, :, c] /= 255.0
            
            img = cv2.resize(img, (rsz_size, rsz_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img[diff_st:-diff_ed, diff_st:-diff_ed, :]  # center crop
            img = np.swapaxes(img, 0, 2)  # HWC2CWH
            img = np.swapaxes(img, 1, 2)  # CWH2CHW

            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            for c in range(3):
                img[c] = (img[c] - means[c]) / stds[c]
            img = img[np.newaxis, :]

            net.blobs['data'].data[idx] = img

        net_st = time.time()
        out = net.forward()
        net_time += (time.time() - net_st)

        prob = out[out.keys()[0]]
        if len(prob.shape) == 4:
            prob = np.squeeze(prob, axis=(2,3))
        ret = np.argmax(prob, axis=1)
        count += (ret == labels).tolist().count(True)

        if ed % 1000 == 0:
            print('[%d/%d]: %.4f, cost %.4f, forward: %.4f' % (ed, len(gts),
                float(count) / ed, (time.time() - start) / ed, net_time / ed))

    print('top1 acc: %.4f' % (float(count) / len(gts)))
