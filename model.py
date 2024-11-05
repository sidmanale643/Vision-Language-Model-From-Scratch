import torch
import torch.nn as nn
import torch.nn.functional as F
from image_encoder import Args , ImageEncoder , MHA

class RMS_Norm(nn.Module):
    def __init__(self , args):
        super().__init__()
        
        self.eps = 1e-4
        self.g = nn.Parameter(torch.ones(args.d_model))
        
    def forward(self , x):
        norm = torch.sqrt(torch.mean(x ** 2 , dim = -1 , keepdim =  True)) + self.eps 
        normed_val =  x * norm * self.g
        return normed_val
            
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
        
        self.n_layers = args.n_layers
        self.embeddings = nn.Embedding(args.vocab_size , args.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMS_Norm(args)
        
    def forward(self , x):
        
        x = self.embeddings(x)
 
        for layer in self.encoder:
            x = layer(x)
            
        normed_x = self.norm(x)
        return normed_x
            
class EncoderBlock(nn.Module):
    def __init__(self , args):
        super().__init__()
        
        self.ffn = FFN(args)
        self.norm1 = RMS_Norm(args)
        self.norm2 = RMS_Norm(args)
        self.mha = MHA(args , True)
        self.image_encoder = ImageEncoder(args)
        
    def forward(self , img , x):
    
        normed_embeddings = self.norm1(x)
        image_encoder = self.image_encoder(img)
        mha_out = self.mha(normed_embeddings , image_encoder , image_encoder) + normed_embeddings
        
        mha_normed = self.norm2(mha_out)
        ffn_out = self.ffn(mha_normed) + mha_out
        return ffn_out
    
args = Args()
model = Model(args)

print(model)

        
