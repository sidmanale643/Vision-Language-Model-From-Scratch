import torch
import torch.nn as nn

class Args:
    def __init__(self):
        self.patch_size = 16
        self.img_size = 224
        self.n_patches = self.img_size // self.patch_size
        self.d_model = 768
        self.n_heads = 12
        self.n_layers = 12 
        self.n_seq = 1024
        self.vocab_size = 50000
    
class MHA(nn.Module):
    def __init__(self , args ,  decoder):
        super().__init__()
        
        if decoder:
            self.n_seq = args.n_patches * args.n_patches
        else:
            self.n_seq = args.n_seq
            
        self.d_model = args.d_model
        self.d_k = args.d_model // args.n_heads
        self.n_heads = args.n_heads
        self.decoder = decoder
        
        self.w_q = nn.Linear(args.d_model , args.d_model , bias = False)
        self.w_k = nn.Linear(args.d_model , args.d_model , bias = False)
        self.w_v = nn.Linear(args.d_model , args.d_model , bias = False)
        self.w_o = nn.Linear(args.d_model , args.d_model , bias = False)
        
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

    
class ImageEncoder(nn.Module):
    def __init__(self , args):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels = 3 , out_channels = args.d_model , kernel_size = (args.patch_size ,args.patch_size ) , stride = args.patch_size)
        self.pos_embeddings = nn.Parameter(torch.zeros(1 , args.n_patches  * args.n_patches , args.d_model))  #pos_emb -> (1 , 14 , 768)
        self.mha = MHA(args , decoder = True)
        
    def forward(self , x):
        
        #x -> (b , c , h , w) ;           
        #patch embeddings ->  (b , 768 , 14 , 14) ;
        patch_embeddings = self.conv(x)
        
        patch_embeddings = patch_embeddings.flatten(2).transpose(1,2) #->  (b , 14 * 14 , 768) ;

        x = patch_embeddings + self.pos_embeddings
        
        mha_out = self.mha(x , x , x)
        
        return mha_out


 