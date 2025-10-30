import os
import json
import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt

class HollaRecSystem(BaseSystem):
    def __init__(
        self, opt,
        train_dataloader, valid_dataloader, test_dataloader,
        vocab, side_data=None,
        restore_system=False, interact=False, debug=False
    ):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.

        """
        super(HollaRecSystem, self).__init__(
            opt, train_dataloader, valid_dataloader, test_dataloader,
            vocab, side_data, restore_system, interact, debug)
        
        self.ind2tok = vocab["ind2tok"]
        self.end_token_idx = vocab["tok2ind"]["</s>"]
        
        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        # rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 100, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, label in zip(rec_ranks, item_label):
            self.evaluator.rec_evaluate(rec_rank, label)

    def conv_evaluate(self, prediction, response, batch_user_id=None, batch_conv_id=None):
        prediction = prediction.tolist()
        response = response.tolist()
        if batch_user_id is None:
            for p,r in zip(prediction, response):
                p_str = ind2txt(self.ind2tok, p, self.end_token_idx)
                r_str = ind2txt(self.ind2tok, r, self.end_token_idx)
                self.evaluator.gen_evaluate(p_str, [r_str], p)
        else:
            for p, r, u_id, c_id in zip(prediction, response, batch_user_id, batch_conv_id):
                p_str = ind2txt(self.ind2tok, p, self.end_token_idx)
                r_str = ind2txt(self.ind2tok, r, self.end_token_idx)
                self.evaluator.gen_evaluate(p_str, [r_str], p)

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        if stage=='rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['movie'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode!='test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode=='train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)
                self.conv_evaluate(preds, batch['response'], batch.get('user_id', None), batch.get('conv_id', None))

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f"[Recommendation epoch {str(epoch)}]")
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')

            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')

                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric, 'rec'):
                    break
            
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversational(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f"[Conversational epoch {str(epoch)}]")
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')

            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(self.conv_batch_size):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')

                # early stop
                metric = self.evaluator.optim_metrics['conv_loss']
                if self.early_stop(metric, 'conv'):
                    break
            
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(self.conv_batch_size):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversational()
    
    def interact(self):
        pass