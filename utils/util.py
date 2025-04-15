import torch
import torch.nn as nn
from args import args
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_res(model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep = model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels


def cls_logi(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )

    pipe.fit(features, y)
    return pipe


class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(outputs, labels)
        return loss


class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)
        return align_loss

def lr_warmup(current_epoch):
    warmup_epochs = 0.1 * args.num_epoch
    if current_epoch < warmup_epochs:
        return current_epoch / warmup_epochs
    else:
        return -(1-0.01) / (args.num_epoch - warmup_epochs) * (current_epoch-warmup_epochs) + 1