import os
import os.path
import json
import numpy as np
import sys
import argparse
import pickle
import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def arg_parser():
    parser = argparse.ArgumentParser(description='Convert .npz to .ply')
    parser.add_argument('-i', '--input', type=str, help='project path')
    parser.add_argument('-o', '--output', type=str, default='output', help='output path')
    parser.add_argument('-t', '--train', action='store_true', help='train or not')
    arg = parser.parse_args()

    return arg


class Npz2Ply():
    def __init__(self, input_path, output_path, train=True):
        self.train = train
        self.input_path = input_path
        self.output_path = output_path
        self.datapath = os.listdir(self.input_path)
        if train:
            self.datapath = [d for d in self.datapath if int(d.split('_')[1].split('s')[1]) <= 5]
        else:
            self.datapath = [d for d in self.datapath if int(d.split('_')[1].split('s')[1]) > 5]
        self.datapath = [d.split('.')[0] for d in self.datapath]
        for idx in range(len(self.datapath)):
            self.load_npz(idx)
            self.write_ply(idx)

    def load_npz(self, idx): # takes about 5G memory to load
        filename = self.datapath[idx]
        print('Loading: {}'.format(filename))
        result = np.load(os.path.join(self.input_path, filename+'.npz'))
        self.data = result['point_clouds']
        print("Frame Number: {}".format(len(self.data)))

    def write_ply(self,idx,text=True):
        save_root = os.path.join(self.output_path, self.datapath[idx])
        os.makedirs(save_root,exist_ok=True)
        filename = self.datapath[idx]
        print("Saving: {}".format(filename))
        for j in range(len(self.data)):
            points = self.data[j]
            save_path = os.path.join(save_root, filename+'_frame_{}.ply'.format(j))
            if os.path.exists(save_path):
                continue
            points = [(points[k, 0], points[k, 1], points[k, 2]) for k in range(points.shape[0])]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
            PlyData([el], text=text).write(save_path)


if __name__=="__main__":
    args = arg_parser()
    d = Npz2Ply(input_path=args.input, output_path=args.output, train=args.train)
    print('Done!')