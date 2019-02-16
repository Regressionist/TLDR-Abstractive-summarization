import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device=torch.device('cuda')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder,self).__init__()
        
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.gru=nn.GRU(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        #self.linear=nn.Linear(2*self.hidden_size,hidden_size)
    
    def forward(self,enc_input,device=device):
        #enc_input: batch*seq_len
        seq_len=enc_input.size(1)
        batch_size=enc_input.size(0)
        hidden=torch.zeros(2, batch_size, self.hidden_size, device=device).detach()
        encoded=torch.zeros(batch_size, seq_len, 2*self.hidden_size, device=device).detach()
        for i in range(seq_len):
            embedded = self.embedding(enc_input[:,i]) #embedded: batch*emb_dim
            encoder_outputs, hidden = self.gru(embedded.unsqueeze(1),hidden)
            #enc_outpus: batch,1,2*hidden_dim
            #hidden:2,batch,hidden
            encoded[:,i]=encoder_outputs.view(-1,2*self.hidden_size)
        return encoded, hidden
    
class Decoder(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, max_len=400):
        super(Decoder,self).__init__()
        
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.max_len=max_len
        
        self.attn=nn.Linear(self.hidden_dim+self.embedding_dim,self.max_len)
        self.attn_combine=nn.Linear(self.hidden_dim+self.embedding_dim,self.embedding_dim)
        self.linear1=nn.Linear(2*self.hidden_dim,hidden_dim)
        self.linear2=nn.Linear(2*self.hidden_dim,hidden_dim)
        self.embedding=nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru=nn.GRU(self.embedding_dim, self.hidden_dim,num_layers=1, batch_first=True)
        self.linear=nn.Linear(self.hidden_dim, self.vocab_size)
        
        
    def forward(self,enc_output, enc_hidden, dec_input, dec_len, target,val=False,device=device):
        #enc_output: batch,seq_len,2*hidden_dim
        #enc_hidden:batch,2,hidden
        #dec_input:batch,1
        #target:batch,dec_len
        batch_size=enc_hidden.size(1)
        enc_seq_len=enc_output.size(1)
        enc_hidden=(self.linear1(enc_hidden.view(-1,2*self.hidden_dim))) #enc_hidden:batch,hidden_dim
        enc_output=(self.linear2(enc_output.view(-1,2*self.hidden_dim))).view(batch_size,-1,self.hidden_dim) #enc_output:batch,enc_seq_len,hidden_dim
        dec_hidden=enc_hidden.unsqueeze(0)
        decoded=torch.zeros(batch_size, dec_len, device=device).detach()
        decoded_outputs=torch.zeros(batch_size, dec_len, self.vocab_size, device=device).detach()
        r=random.random()
        
        for i in range(dec_len):
            embedded=self.embedding(dec_input) #embedded:batch,1,emb_dim
            attn_weights=F.softmax(self.attn(torch.cat((enc_hidden,embedded.view(-1,self.embedding_dim)),dim=1)),dim=1).unsqueeze(1)
            attn_applied=torch.bmm(attn_weights,enc_output) #attn_applied:b,1,hidden_dim
            output=torch.cat((embedded.view(-1,self.embedding_dim),attn_applied.view(-1,self.hidden_dim)),dim=1)
            output = self.attn_combine(output) #output: b,emb_dim
            output=F.relu(output)
            output, dec_hidden = self.gru(output.unsqueeze(1), dec_hidden) #output:b,1,hidden_dim
            output=F.log_softmax(self.linear(output.view(-1,self.hidden_dim)),dim=1)#output:b,1,vocab
            topv, topi = output.view(-1,self.vocab_size).topk(1)
            if r<0.5 or val==True: # don't use teacher forcing
                dec_input = topi.detach()
            else:  #Use teacher forcing
                dec_input=target[:,i].view(batch_size,1)
            decoded[:,i]=topi.detach().squeeze(1)
            decoded_outputs[:,i,:]=output
        return decoded, dec_hidden, decoded_outputs
        
