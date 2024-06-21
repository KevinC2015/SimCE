# SimCE


Implementation of the paper "SimCE: Simplifying Cross-Entropy Loss for Collaborative Filtering".

This work investigates the different loss functions in collaborative filtering (CF) from the perspective of  multiple negative sample. The proposed SimCE provides a new learning objective for CF-based recommender systems, which directly optimizes the hardest negative samples. A simple MF/LightGCN encoder optimizing this loss can achieve superior performance compared to other loss function. 

## Training with SimCE

This learning objective is easy to implement as follows (PyTorch-style):

```python
def bpr(user_emb, pos_item_emb, neg_item_emb, gamma=1e-5):
    # user_emb: [batch, dim]
    # pos_item_emb: [batch, dim]
    # neg_item_emb: [batch, dim]
    
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score))
    
    return torch.mean(loss)

    
def ssm(user_emb, pos_item_emb, neg_item_emb):
    # user_emb: [batch, dim]
    # pos_item_emb: [batch, dim]
    # neg_item_emb: [batch, num_neg, dim]
    
    num_neg, dim  = neg_item_emb.shape[1], neg_item_emb.shape[2]
    neg_item_emb = neg_item_emb.reshape(-1, num_neg, dim)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)
    loss = torch.log(1 + torch.exp(neg_score - pos_score.unsqueeze(dim=1)).sum(dim=1))
    
    return torch.mean(loss)

    
def simce(user_emb, pos_item_emb, neg_item_emb, margin=5.0):
    # user_emb: [batch, dim]
    # pos_item_emb: [batch, dim]
    # neg_item_emb: [batch, num_neg, dim]
    
    num_neg, dim  = neg_item_emb.shape[1], neg_item_emb.shape[2]
    neg_item_emb = neg_item_emb.reshape(-1, num_neg, dim)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb.unsqueeze(dim=1), neg_item_emb).sum(dim=-1)
    neg_score = torch.max(neg_score, dim=-1).values
    loss = torch.relu(margin - pos_score + neg_score)
    
    return torch.mean(loss)
```

 The datasets used in the paper are mainly collected from other popular repositories:
 1. https://github.com/Coder-Yu/SELFRec/tree/main/dataset
 2. https://github.com/hexiangnan/neural_collaborative_filtering
 3. https://github.com/THUDM/MCNS/tree/master/recommendation/data
 4. https://github.com/reczoo/RecZoo/tree/main/matching/cf/SimpleX

Our framework is mainly built up on the [SELFRec](https://github.com/Coder-Yu/SELFRec).


## Run the code

```shell
# Movielean
python3 main.py --dataset=ml-1M --trainset=./dataset/ml-1M/train.txt --testset=./dataset/ml-1M/test.txt --model=LightGCN --num_neg=64 --margin=5.0 --loss=simce
```



The main hyper-parameters of SimCE includes:

| Param              | Default | Description                                    |
| ------------------ | ------- | ---------------------------------------------- |
| `--embedding_size` | 64      | The embedding size                             |
| `--gamma`          | 5.0     | the margin in the loss                         |
| `--encoder`        | MF      | The encoder type: MF / LightGCN                |
| `--loss`           | simce    | The loss: bpr / ssm / simce                     |


