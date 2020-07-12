from fastai.text import *
from hparams import Hparams
from data_preprocessing import databuncher
from model import Seq2SeqRNN, Seq2SeqRNN_tf
from bleu import CorpusBLEU
import logging

# creat random embedding and train from scratch
def create_rnd_emb(itos, em_sz=256):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    return emb

# loss function
def seq2seq_loss(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    return CrossEntropyFlat()(out, targ)

# metric0: accuracy
def seq2seq_acc(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    out = out.argmax(2)
    return (out==targ).float().mean()

def rnn_setup(x_itos, y_itos, model_type="rnn_tf"):
    # randomly initialize word embedding
    emb_enc = create_rnd_emb(x_itos)
    emb_dec = create_rnd_emb(y_itos)
    
    # instantiate the model
    if model_type == "rnn_tf":
        rnn_tf = Seq2SeqRNN_tf(emb_enc, emb_dec, 256, 30).cuda()
        return rnn_tf
    else:
        rnn = Seq2SeqRNN(emb_enc, emb_dec, 256, 30).cuda()
        return rnn
    

if __name__ == "__main__":
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    
    # identify the device to use
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using %s" % torch.cuda.get_device_name(0))
    
    # get dataloader from stored databunch
    data = databuncher(True)
    
    # set batch size
    data.batch_size = hp.batch_size
    
    # instantiate RNN
    rnn_tf = rnn_setup(data.x.vocab.itos, data.y.vocab.itos, model_type="rnn_tf")
    
    # instantiate Learner 
    learn = Learner(data, 
                    rnn_tf, 
                    loss_func=seq2seq_loss, 
                    metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))], callback_fns=[callbacks.CSVLogger])
    
    # training sequence
    learn.fit_one_cycle(hp.epochs, 
                        max_lr=hp.lr, 
                        callbacks=[callbacks.SaveModelCallback(learn, every='epoch', monitor='accuracy', name='model')])
    
    