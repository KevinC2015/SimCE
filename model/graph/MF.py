import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss, bpr_k_loss, ccl_loss, directau_loss, simce_loss, ssm_loss
import torch.nn.functional as F
from util.args import get_params
import time
args = get_params()

class MF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MF, self).__init__(conf, training_set, test_set)
        self.model = Matrix_Factorization(self.data, self.emb_size)
        if args.loss in ['ssm', 'simce']:
            self.maxEpoch = 50
            

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            t0 = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                reg_loss = l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size

                

                
                #batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size

                if args.loss == 'bpr':
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'ssm':
                    rec_loss = ssm_loss(user_emb, pos_item_emb, neg_item_emb)
                elif args.loss == 'simce':
                    rec_loss = simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=args.margin)
                    
                
                batch_loss = rec_loss + reg_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                # if n % 100==0 and n>0:
                #     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % 1 == 0:
                self.fast_evaluation(epoch)
            print('each epoch: {} seconds'.format(time.time() - t0))
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            user_emb = self.user_emb[u]
            item_emb = self.item_emb
            if args.loss in ['directau']:
                user_emb = F.normalize(user_emb, dim=-1)
                item_emb = F.normalize(item_emb, dim=-1)
            score = torch.matmul(user_emb, item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })

        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']


