import numpy as np
import pandas as pd
import spacy
import operator
import os
import torch
import glob

class Corpus(object):
    def __init__(self, path='data/cnn_tokenized/', vocab_path='processed_data/vocab', max_size=50000, first_run=False, max_enc_len=400, max_dec_len=100):
        self.path=path
        self.max_enc_len=max_enc_len
        self.max_dec_len=max_dec_len
        self.nlp = spacy.load('en')
        self.UNK_TOKEN='<unk>'
        self.PAD_TOKEN='<pad>'
        self.EOS_TOKEN='<eos>'
        self.SOS_TOKEN='<sos>'
        self.w2i={}
        self.i2w={}
        self.count=0
        self.counter=0
        self.max_size=max_size
        rands=[50262, 60514, 59047, 31609, 18888, 22674, 43177, 81223, 56843, 90315, 52224,  5701, 81411, 31190, 81189, 17617, 62764, 41486, 8403, 63411, 23366, 71045, 48061, 30457,  8942, 10114, 73604, 26995,  7058, 52211, 78931, 18414, 20849, 48624, 83757, 75600, 34298, 71987, 52232, 57624,  3018, 82771, 13776, 42617, 52178, 78123, 26319, 26481, 11163, 64949]
        for w in ['<unk>','<pad>','<eos>','<sos>']:
            self.w2i[w]=self.count
            self.i2w[self.count]=w
            self.count+=1
        with open(vocab_path) as file:
            for line in file:
                w=line.split()
                if w[0] not in self.w2i:
                    #print(w[0])
                    self.w2i[w[0]]=self.count
                    self.i2w[self.count]=w[0]
                    self.count+=1
                    if (self.max_size<=self.count):
                        break
        
        if(first_run==True):
            file_list=glob.glob(path+'*.story')
            len_dic={}
            for file in file_list:
                with open(file) as f:
                    text=f.read()
                text = text.split('@highlight')
                body = text[0].strip()
                len_dic[file]=len(body.split())
            len_dic=sorted(len_dic.items(),key=operator.itemgetter(1))
            self.df=pd.DataFrame(len_dic, columns=['file_path','body_len'])
            sum_lens=[]
            for file in file_list:
                with open(file) as f:
                    text=f.read()
                text = text.replace("\n\n",' ')
                text = text.replace("  ",' ')
                text = text.lower()
                text = text.split('@highlight')
                body = text[0].strip()
                summaries = text[1:]
                sum_out = []
                for summary in summaries:
                    summary = summary.strip()
                    summary = summary + ' .'
                    sum_out.append(summary)
                    summaries = ' '.join(sum_out)
                summaries=summaries.split()
                sum_lens.append(len(summaries))
            self.df['sum_len']=sum_lens
            self.df=self.df[self.df['body_len']!=0]
            self.df.to_csv('lengths.csv',index=False)
            
            self.df_val=self.df.iloc[rands]
            self.df_val=self.df_val.reset_index(drop=True)
            self.df_train=self.df.drop(rands,axis=0)
            self.df_train=self.df_train.reset_index(drop=True)
        else:
            self.df=pd.read_csv('lengths.csv')
            self.df_val=self.df.iloc[rands]
            self.df_val=self.df_val.reset_index(drop=True)
            self.df_train=self.df.drop(rands,axis=0)
            self.df_train=self.df_train.reset_index(drop=True)
        
    def get_train_minibatch(self):
        body_batch=[]
        summary_batch=[]
        #max_length_body=max(self.df_train['body_len'][self.counter:self.counter+20])
        max_length_summary=max(self.df_train['sum_len'][self.counter:self.counter+20])
        max_length_body=400
        #max_length_summary=100
        for index, row in self.df_train.iterrows():
            if (index>=self.counter and index<self.counter+20):
                with open(row['file_path']) as f:
                    text=f.read()
                    text = text.replace("\n\n",' ')
                    text = text.replace("  ",' ')
                    text = text.lower()
                    text = text.split('@highlight')
                    body = text[0].strip()
                    body=body.split()
                    
                    if len(body)>self.max_enc_len:
                        body=body[:self.max_enc_len]
                    summaries = text[1:]
                    sum_out = []
                    for summary in summaries:
                        summary = summary.strip()
                        summary = summary + ' .'
                        sum_out.append(summary)
                        summaries = ' '.join(sum_out)
                    summaries=summaries.split()
                    if len(summaries)>self.max_dec_len:
                        summaries=summaries[:self.max_dec_len]
                    
                    
                    body=[self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in body]
                    summary=[self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in summaries]
                    summary.append(self.w2i['<eos>'])
                    while len(body)<max_length_body:
                        body.append(self.w2i['<pad>'])
                    while len(summary)<max_length_summary:
                        summary.append(self.w2i['<pad>'])
                #print (np.array(body).shape)
                if (np.array(body).shape==(max_length_body,) and np.array(summary).shape==(max_length_summary,)):
                    body_batch.append(body)
                    summary_batch.append(summary)
        
        body_batch=np.matrix(body_batch)
        summary_batch=np.matrix(summary_batch)
        if (self.counter+20<self.df_train.shape[0]):
            self.counter=self.counter+20
        else:
            self.counter=0
        
        return torch.from_numpy(body_batch), torch.from_numpy(summary_batch)
    
    def get_validation_batch(self):
        #body_batch=[]
        #summary_batch=[]
        max_length_body=400
        max_length_summary=max(self.df_val['sum_len'][:])
        body_batch=np.zeros((44,max_length_body))
        summary_batch=np.zeros((44,max_length_summary))
        c=0
        for index, row in self.df_val.iterrows():
            with open(row['file_path']) as f:
                text=f.read()
                text = text.replace("\n\n",' ')
                text = text.replace("  ",' ')
                text = text.lower()
                text = text.split('@highlight')
                body = text[0].strip()
                body=body.split()
                if len(body)>self.max_enc_len:
                    body=body[:self.max_enc_len]
                summaries = text[1:]
                sum_out = []
                for summary in summaries:
                    summary = summary.strip()
                    summary = summary + ' .'
                    sum_out.append(summary)
                    summaries = ' '.join(sum_out)
                summaries=summaries.split()
                if len(summaries)>self.max_dec_len:
                    summaries=summaries[:self.max_dec_len]
                
                body=[self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in body]
                summary=[self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in summaries]
                if len(summary)<self.max_dec_len:
                    summary.append(self.w2i['<eos>'])
                while len(body)<max_length_body:
                    body.append(self.w2i['<pad>'])
                while len(summary)<max_length_summary:
                    summary.append(self.w2i['<pad>'])
            try:
                body_batch[c]=body
                summary_batch[c]=summary
                c=c+1
            except:
                c=c
            
        return torch.from_numpy(body_batch), torch.from_numpy(summary_batch)