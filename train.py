import torch
import warnings

warnings.filterwarnings('ignore')
from utils.args import args, Train_data_all, Train_data, Val_data, Test_data
from utils.dataset import Dataset
from model.NANOSSL import NANOSSL
from trainer import Trainer
import torch.utils.data as Data


def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    all_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    all_loader = Data.DataLoader(all_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = all_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    val_loader = None
    if Val_data is not None:
        val_dataset = Dataset(device=args.device, mode='val', data=Val_data, wave_len=args.wave_length)
        val_loader = Data.DataLoader(val_dataset, batch_size=args.test_batch_size)

    model = NANOSSL(args)
    print(model)

    print('model initial ends')
    trainer = Trainer(args, model, all_loader, train_linear_loader, val_loader, test_loader, verbose=True)

    trainer.pretrain()
    trainer.finetune()


if __name__ == '__main__':
    main()
