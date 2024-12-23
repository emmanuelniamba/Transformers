   
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
def scaled_dot_product(q,k,v,):
    # before all we calculate the dot product between the query and the key
    dim_of_k=k.size(-1) # we get the size of the last dimension of the key matrix
    attention_score=torch.bmm(q,k.transpose(1,2))/sqrt(dim_of_k)# bmm is a batch matrix multiplication and trasnpose(1,2) is used to swap the second and third dimension of the tensor, for example if we have a tensor of shape (2,3,4) after applying transpose(1,2) we will have a tensor of shape (2,4,3)
    # we can have  really big values for the dot product, so we  will now us Foftmax function to normalize the values,and we will have a probability distribution ans we apply it to the last dimension of the tensor
    score=F.softmax(attention_score,dim=-1)
    # we now apply the softmax to the value matrix
    attention_output=torch.bmm(score,v)
    return attention_output
class Attentionhead(nn.Module): #nn.Module is the base class for all neural network modules in PyTorch
    def __init__(self,dim_of_emb,dim_of_head):
        super().__init__()
        # q is the query matrix,k the key matrix,v the values matrix, it allows us to apply one linear transformation to the input witj their own set of learnable parameters 
        self.q=nn.Linear(dim_of_emb,dim_of_head) 
        self.k=nn.linear(dim_of_emb,dim_of_head)
        self.v=nn.linear(dim_of_emb,dim_of_head)
    def forward(self,hidden_state):
        # we apply the linear transformation to the input
        query=self.q(hidden_state) 
        key=self.k(hidden_state)
        value=self.v(hidden_state)
        attention_output=scaled_dot_product(query,key,value)
        return attention_output
    
# now we will implement the multihead attention, the particular here is to allow the model to focus on differents patterns in the inputs by using multiple attention heads
#For exemple for a model like Bert where the total dimensions is 768 we use 12  heads so we will have 12 different group of (query,key,value) matrices, in these matixes we will have 64 dimensions for each head, we use 64 because 768/12=64 for have same computation for each head 

#from transformers import AutoConfig  
#config = AutoConfig.from_pretrained(model_ckpt)
class MultiheadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        dim_of_emb=config.hidden_size # the dimension of the input
        num_of_head=config.num_attention_heads # the number of heads
        dim_of_head=dim_of_emb//num_of_head # the dimension of each head
        self.heads=nn.ModuleList([Attentionhead(dim_of_emb,dim_of_head) for _ in range(num_of_head)]) # we create a list of attention heads num_of_head times
        self
        
        
    