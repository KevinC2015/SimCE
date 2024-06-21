import argparse
def get_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="LightGCN")
    parser.add_argument('--num_neg', type=int, default=10)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--loss', type=str, default='bpr')
    parser.add_argument('--trainset', type=str, default="./dataset/yelp2018/train.txt")
    parser.add_argument('--testset', type=str, default="./dataset/yelp2018/test.txt")
    


    args, _ = parser.parse_known_args()
    
    return args