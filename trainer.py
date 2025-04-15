import time
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from utils.util import CE, Align, lr_warmup, cls_logi, get_res
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import pdb


class Trainer():
    def __init__(self, args, model, all_loader, train_loader, val_loader, test_loader, verbose=False):
        
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.all_loader = all_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain

        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.model = model.to(torch.device(self.device))
        self.current_time = time.strftime("%Y%m%d", time.localtime())

        self.cr = CE(self.model)
        self.test_cr = torch.nn.CrossEntropyLoss()
        self.step = 0
        self.best_metric = -1e9
        self.best_metric_test = -1e9
        self.metric = 'acc'

        self.model_param = str(round(self.args.mask_ratio, 1)) + '_' + str(self.args.wave_length) + '_' + str(self.args.d_model) + '_' + \
            str(self.args.layers) + '_' + str(self.args.num_epoch_pretrain)
        self.model_param_ft = str(round(self.args.mask_ratio, 1)) + '_' + str(self.args.wave_length) + '_' + str(self.args.d_model) + '_' + \
            str(self.args.layers) + '_' + str(self.args.num_epoch)

    def pretrain(self):
        self.fname = './log/' + self.args.dataset + '/pretrain_'
        if os.path.exists(self.fname):
            for i in range(10000):
                fname2 = self.fname + str(i)
                if not os.path.exists(fname2):
                    self.fname = fname2
                    break
        print("save log in: ", self.fname)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = None
        eval_acc = 0
        align = Align()
        
        self.model.copy_weight()
        self.model.eval()
        train_rep, train_label = get_res(self.model, self.train_loader)
        test_rep, test_label = get_res(self.model, self.test_loader)
        clf = cls_logi(train_rep, train_label)
        acc = clf.score(test_rep, test_label)
        print('test acc: ', acc)

        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.all_loader)
            loss_mse = 0
            if self.scheduler:
                self.scheduler.step()
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction] = self.model.forward_pretrain(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction) 
                loss_mse += align_loss.item()
                loss = align_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            if (epoch + 1) % 5 == 0:
                print('pretrain epoch {0}\nloss {1:.3f}\n'.format(epoch + 1, loss_mse / (idx + 1)))
                self.model.eval()
                train_rep, train_label = get_res(self.model, self.train_loader)
                test_rep, test_label = get_res(self.model, self.test_loader)
                clf = cls_logi(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print('test acc: ', acc)
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_' + self.fname.split("_")[-1] + '_model_' + self.model_param + '_' + self.current_time + '.pkl')

    def finetune(self):
        print('finetune')
        self.fname_ft = './log/' + self.args.dataset + '/finetune_'
        if os.path.exists(self.fname_ft):
            for i in range(10000):
                fname2 = self.fname_ft + str(i)
                if not os.path.exists(fname2):
                    self.fname_ft = fname2
                    break
        print("save log in: ", self.fname_ft)
        if self.args.load_pretrained_model:
            print('load pretrained model')
            if self.args.model_id:
                state_dict = torch.load('exp/ONT/test/' + self.args.model_id + '.pkl', map_location=self.device)
            else:
                state_dict = torch.load(os.path.join(self.save_path, 'pretrain_' + self.fname.split("_")[-1] + '_model_' + self.model_param + '_' + self.current_time  + '.pkl'), map_location=self.device)
            params_train = [
                            'predict_head.weight', 'predict_head.bias',
                            'pred_heads.linear1.weight', 'pred_heads.linear1.bias', 'pred_heads.linear2.weight', 'pred_heads.linear2.bias', 'pred_heads.linear3.weight', 'pred_heads.linear3.bias']
            try:
                self.model.load_state_dict(state_dict)
            except:
                print("\nWrong\n")
                model_state_dict = self.model.state_dict()
            keys = list(state_dict.keys())
            for pretrain in keys:
                if pretrain in params_train:
                    state_dict.pop(pretrain)
            self.model.load_state_dict(state_dict, strict=False)
            print("\nSuccessfully loaded:{}\n".format(self.args.model_id if self.args.model_id else self.fname.split("_")[-1]))
        
        self.model.linear_proba = False
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.num_epoch, eta_min=0.01 * self.args.lr)

        self.result_file = open(self.save_path + '/result.txt', 'a+')

        for epoch in range(self.num_epoch):
            loss_epoch, val_metric, test_metric, time_cost = self._train_one_epoch()
            if (epoch+1) % 10 == 0:
                print('Finetune epoch:{0}, loss:{1:.3f}, training_time:{2:.3f}'.format(epoch + 1, loss_epoch, time_cost))

        print('finetune_' + self.fname_ft.split("_")[-1] + ', the best acc of {}: {:.3f}'.format(self.args.dataset, self.best_metric_test), '\n',
                'the final acc of {}:{:.3f}'.format(self.args.dataset, test_metric['acc']), '\n',
                'the final f1 of {}:{:.3f}'.format(self.args.dataset, test_metric['f1']), file=self.result_file)
        self.result_file.close()

        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)  # fine-tune model + 分类头
            # loss = self.fl(batch)
            loss_sum += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        self.step += 1
        if self.scheduler:
            self.scheduler.step()
        if self.val_loader is not None:
            val_metric = self.test_model()
        else:
            val_metric = {'acc': 0, 'val_loss': 0}
        test_metric = self.test_model()
        print(test_metric)
        if test_metric[self.metric] > self.best_metric_test:
            torch.save(self.model.state_dict(), self.save_path + '/finetune_' + self.fname_ft.split("_")[-1] + '_model_' + self.model_param_ft + '.pkl')
            print('saving model of step{0}'.format(self.step), file=self.result_file)
            self.best_metric_test = test_metric[self.metric]
        if val_metric[self.metric] > self.best_metric:
            self.best_metric = val_metric[self.metric]
            self.best_metric_val_test = test_metric[self.metric]
            self.best_step = self.step
        self.model.train()

        return loss_sum / (idx + 1), val_metric, test_metric, time.perf_counter() - t0


    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        confusion_mat = self._confusion_mat(label, pred)
        print(confusion_mat)
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        for k, v in metrics.items():
            metrics[k] = round(v, 3)
        return metrics

    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2))
        _, pred = torch.topk(scores, 1)
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
            # 混淆矩阵mat转换为概率值
        mat = mat / np.sum(mat, axis=1, keepdims=True)
        mat = np.round(mat, 2)
        return mat

