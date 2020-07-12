import argparse

class Hparams():
    """
    hyperparameter passing
    """
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument("--corpus", 
                        default=r"C:\Users\zhimi\.fastai\data\giga-fren\questions_easy.csv", 
                        help="path of parallel corpus csv file")
    
    # validation set store path
    parser.add_argument("--valid_path", 
                        default="./data/valid.csv", 
                        help="path of validation set csv file")
    
    # databunch pkl path
    parser.add_argument("--db_path", 
                        default="./data/", 
                        help="path of databunch pkl file")
    
    # training scheme
    parser.add_argument("--batch_size",
                        default=4,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs",
                        default=2,
                        help="number of epochs")
    parser.add_argument("--ckpt",
                        default="./data/models",
                        help="training directory for training result")
    
    # test scheme    
    parser.add_argument("--valid_result",
                        default="./data/valid_result.csv",
                        help="csv for validation result")