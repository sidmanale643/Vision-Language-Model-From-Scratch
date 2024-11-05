from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Args:
    def __init__(self , d_model , vocab_size , n_seq , n_layers):
        self.d_model = d_model
        self.n_seq = n_seq
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        
class RMS_Norm(nn.Module):
    def __init__(self , Args):
        super().__init__()
        
        self.eps = 1e-4
        self.g = torch.randint(torch.ones(Args.d_model))
        
    def forward(self , x):
        norm = torch.sqrt(torch.mean(x ** 2 , dim = -1 , keepdim =  True)) + self.eps 
        normed_val =  x * norm * self.g
        return normed_val
    
class MHA(nn.Module):
    def __init__(self , args ,  decoder):
        super().__init__()
        
        self.n_seq = args.n_patches * args.n_patches
        self.d_model = args.d_model
        self.d_k = args.d_model // args.n_heads
        self.n_heads = args.n_heads
        self.decoder = decoder
        
        self.w_q = nn.Linear(args.d_model , args.d_model)
        self.w_k = nn.Linear(args.d_model , args.d_model)
        self.w_v = nn.Linear(args.d_model , args.d_model)
        self.w_o = nn.Linear(args.d_model , args.d_model)
        
    def forward(self , Q , K , V ):
        
        b_size , n  = Q.size(0) , Q.size(1)
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(b_size , self.n_seq , self.n_heads , self.d_k).transpose(1,2)
        K = K.view(b_size , self.n_seq , self.n_heads , self.d_k).transpose(1,2)
        V = V.view(b_size , self.n_seq , self.n_heads , self.d_k).transpose(1,2)
        
        attention_scores = Q @ K.transpose(2,3) / torch.sqrt(torch.tensor(self.d_k))
        
        if self.decoder:
            mask = torch.randn(n , n).bool().unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(mask == 0 , -1e9)

        attention_weights = torch.softmax(attention_scores , dim = -1)
        
        attention_out = attention_weights @ V
        
        attention_out = attention_out.transpose(1,2).contiguous().view(b_size , self.n_seq , self.d_model)
        
        final_out = self.w_o(attention_out)
        return final_out 
        
class FFN(nn.Module):
    def __init__(self , args):
        super().__init__()
        
        self.fc1 = nn.Linear(args.d_model , 4 * args.d_model , bias = False)
        self.fc2 = nn.Linear(4 * args.d_model , args.d_model , bias = False)
        self.fc3 = nn.Linear(args.d_model , args.d_model , bias = False)
        self.silu = nn.SiLU()   
             
    def forward(self , x):
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))
    
class Model(nn.Module):
    def __init__(self , args):
        super().__init__()

        self.embeddings = nn.Embedding(args.vocab_size , args.d_model)
        self.encoder = nn.ModuleList([Encoder(args) for _ in range(args.n_layers)])
        self.ffn = FFN(args)
        self.norm1 = RMS_Norm(args)
        self.norm2 = RMS_Norm(args)
        self.mha = MHA(args , True)
        
    def forward(self , x):
        

class Encoder(nn.Module):
    def __init__(self , args):
        super().__init__()
        
        pass
        
        