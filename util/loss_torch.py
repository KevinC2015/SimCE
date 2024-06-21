import torch
import torch.nn.functional as F
import torch.nn as nn
from util.args import get_params

args = get_params()

def bpr_loss(user_emb, pos_item_emb, neg_item_emb, gamma=1e-5):
    '''
    user_emb: [batch, dim]
    pos_item_emb: [batch, dim]
    neg_item_emb: [batch, dim]
    '''
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def bpr_k_loss(user_emb, pos_item_emb, neg_item_emb, gamma=1e-5):
    '''
    user_emb: [batch, dim]
    pos_item_emb: [batch, dim]
    neg_item_emb: [batch, num_negs, dim]
    '''
    
    dim = user_emb.shape[-1]
    neg_item_emb = neg_item_emb.reshape(-1, args.num_neg, dim)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)

    loss = -torch.log(gamma + torch.sigmoid(pos_score.unsqueeze(-1) - neg_score))
    
    return torch.mean(loss)

def ssm_loss(user_emb, pos_item_emb, neg_item_emb):
    '''
    user_emb: [batch, dim]
    pos_item_emb: [batch, dim]
    neg_item_emb: [batch, num_negs, dim]
    '''

    dim = user_emb.shape[-1]
    neg_item_emb = neg_item_emb.reshape(-1, args.num_neg, dim)
    
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)
    
    loss = torch.log(1 + torch.exp(neg_score - pos_score.unsqueeze(dim=1)).sum(dim=1))
    return torch.mean(loss)


def ccl_loss(user_emb, pos_item_emb, neg_item_emb, margin=0, weight=None):

    dim = user_emb.shape[-1]
    neg_item_emb = neg_item_emb.reshape(-1, args.num_neg, dim)
    
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)

    

    pos_loss = torch.relu(1 - pos_score)
    neg_loss = torch.relu(neg_score - margin)

    if weight:
        loss = pos_loss + neg_loss.mean(dim=-1) * weight
    else:
        loss = pos_loss + neg_loss.mean(dim=-1)

    return torch.mean(loss)


def directau_loss(user_emb, pos_item_emb, neg_item_emb, gamma=1.0):
    
    align = (user_emb - pos_item_emb).norm(p=2, dim=1).pow(2).mean()
    uniform_user = torch.pdist(user_emb, p=2).pow(2).mul(-2).exp().mean().log()
    uniform_item = torch.pdist(pos_item_emb, p=2).pow(2).mul(-2).exp().mean().log()
    loss = align + gamma * (uniform_user + uniform_item)
    
    return loss


def simce_loss(user_emb, pos_item_emb, neg_item_emb, margin=1.0):

    dim = user_emb.shape[-1]
    neg_item_emb = neg_item_emb.reshape(-1, args.num_neg, dim)
    
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)
    neg_score = torch.max(neg_score, dim=-1).values

    loss = torch.relu(margin - pos_score + neg_score)

    return torch.mean(loss)
    



    

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


#this version is from recbole
def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

