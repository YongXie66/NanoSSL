import argparse
import os
import json
from utils.datautils import *
import pandas as pd

parser = argparse.ArgumentParser()
# model args
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--eval_per_steps', type=int, default=1)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--wave_length', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.6)
parser.add_argument('--reg_layers', type=int, default=8)
# dataset args
parser.add_argument('--save_path', type=str, default='exp/amyloid/test')
parser.add_argument('--dataset', type=str, default='amyloid')
parser.add_argument('--mutation', type=str, default='E22G')
parser.add_argument('--UCR_folder', type=str, default='PhonemeSpectra')
parser.add_argument('--data_path', type=str,
                    default='data/amyloid/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--cv_split', type=int, default=3)
parser.add_argument('--train_ratio', type=float, default=1)

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=300)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--load_pretrained_model', type=int, default=1)
parser.add_argument('--model_id', type=str, default='')

args = parser.parse_args()

if 'ont' in args.data_path:
    path = args.data_path
    Train_data_all, Train_data, Val_data, Test_data = load_ONT(path)
    args.num_class = len(set(Train_data[1]))
elif 'amyloid' in args.data_path:
    path = args.data_path
    # Train_data_all, Train_data, Val_data, Test_data = load_amyloid(path, ['Native', 'Scrambled'])  # 'Scrambled', 'E22G', 'G37R', 'S8Gly', 'S26Gly', 'S26PO4'
    Train_data_all, Train_data, Val_data, Test_data = load_amyloid_CVsplit(path, ['Native', args.mutation], args.cv_split, args.train_ratio)  # 'Native1_40', 'Scrambled', 'E22G', 'G37R', 'S8Gly', 'S26Gly', 'S26PO4'
    args.num_class = len(set(Train_data[1]))

df_class = pd.DataFrame(Train_data_all[1], columns=['class'])

args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
args.lr_decay_steps = args.eval_per_steps
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
