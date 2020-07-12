from hparams import Hparams
from fastai.text import *
import pandas as pd
from data_module import *
from utils import valid2csv

# create a DataBunch (train_loader, valid_loader) from TextList (src)
# if it does not exist
def databuncher(exist:bool=True):
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    if exist:
        data = load_data(hp.db_path, num_workers=0)
        return data
    else:
        df = pd.read_csv(hp.corpus)[:10000]
        df['en'] = df['en'].apply(lambda x:x.lower())
        df['fr'] = df['fr'].apply(lambda x:x.lower())
        # split_by_rand_pct(0.2) the float indicates the percentage of validation set
        src = Seq2SeqTextList.from_df(df, cols='fr').split_by_rand_pct(0.2).label_from_df(cols='en', label_cls=TextList)
        # store the validation set to csv
        valid2csv(src.valid, hp.valid_path)
        # create a DataBunch (train_loader, valid_loader) from TextList (src)
        data = src.databunch(num_workers=0)
        data.save(hp.db_path + "./data_save.pkl")
        return data
        

if __name__ == "__main__":
    data = databuncher(True)
    
    
    
    