import torch
import torch.nn as nn
from model import MHA

class Args:
    def __init__(self):
        self.patch_size = 16
        self.img_size = 224
        self.n_patches = self.img_size // self.patch_size
        self.d_model = 768
        self.n_heads = 12
        self.n_layers = 12       
        
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
    
args = Args()

x = torch.randn(64 , 3 , 224 , 224)
print(x.shape)
img = ImageEncoder(args)
out = img(x)
print(out.shape)


 