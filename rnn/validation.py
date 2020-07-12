from fastai.text import *
from utils import latest_ckpt, get_predictions
from hparams import Hparams
from train import *

if __name__ == "__main__":
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    
    # get dataloader from stored databunch
    data = databuncher(True)
    
    # instantiate model
    rnn_tf = rnn_setup(data.x.vocab.itos, data.y.vocab.itos)
    
    # instantiate Learner from model and data
    learn = Learner(data, 
                    rnn_tf, 
                    loss_func=seq2seq_loss, 
                    metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))], callback_fns=[callbacks.CSVLogger])
    
    # retrieve specific checkpoint and load
    # ckpt = latest_ckpt(hp.ckpt)
    learn.load(file="model_1")
    
    # run on validation set
    inputs, targets, outputs = get_predictions(learn)
    df = pd.DataFrame()
    df["input"], df["target"], df["output"]  = inputs, targets, outputs
    df.to_csv(hp.valid_result)
    