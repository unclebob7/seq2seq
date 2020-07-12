# -*- coding: utf-8 -*-

import math
import sys
import os
import glob
import json
import pandas as pd
from fastai.text import *

# store the validation textlist to csv file
def valid2csv(valid_text, store_path):
    xtmp, ytmp = [], []
    for i in range(len(valid_text.x)):
        xtmp.append(valid_text.x[i].text)
        ytmp.append(valid_text.y[i].text)
    valid_df = pd.DataFrame()
    valid_df['fr'] = xtmp
    valid_df['en'] = ytmp
    valid_df.to_csv(store_path)
    
# load latest checkpoint
def latest_ckpt(dir):
    if not os.path.exists(dir): os.makedirs(dir)
    ckpt_ls = glob.glob("%s\*.pth" % (dir))   # the serialized ckpt file end with ".pt"
    if ckpt_ls == []:
        print("None")
        return None
    else:
        last_ckpt = max(ckpt_ls, key=os.path.getctime)
        return last_ckpt
    
def get_predictions(learn, ds_type=DatasetType.Valid):
    learn.model.eval()
    inputs, targets, outputs = [],[],[]
    with torch.no_grad():
        for xb,yb in progress_bar(learn.dl(ds_type)):
            out = learn.model(xb)
            for x,y,z in zip(xb,yb,out):
                inputs.append(learn.data.valid_ds.x.reconstruct(x.cpu()))
                targets.append(learn.data.valid_ds.y.reconstruct(y.cpu()))
                outputs.append(learn.data.valid_ds.y.reconstruct(z.cpu().argmax(1)))
    inputs = [i.text for i in inputs]
    targets = [i.text for i in targets]
    outputs = [i.text for i in outputs]
    return inputs, targets, outputs

