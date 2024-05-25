import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from collections import Counter
import re
import pathlib
import numpy as np
import torch
from datasets import load_dataset

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 5
UNKOWN_WORD = "<unk>"
PAD_TOKEN = "<pad>"

def load_wine_dataset_into_hg_datasets():
    # Load the full dataset
    datasets_folder = pathlib.Path(r"C:\Users\amrul\programming\deep_learning\dl_projects\Generative_Deep_Learning_2nd_Edition\data")
    wine_review_filepath=datasets_folder/"wine_reviews"/"winemag-data-130k-v2.json"
    return load_dataset(str(wine_review_filepath.parent),'json')
    

def prepare_text_with_country_variety(batch):
    text_with_country_variety = [f"{country} : {province} : {variety} : {description}" for country, province, variety, description in zip(batch['country'],batch['province'], batch['variety'], batch['description'])]
    return {"text": text_with_country_variety}

def get_wine_ds_with_country_variety(wine_data):
    return wine_data.map(prepare_text_with_country_variety,batched=True, batch_size=None)

def custom_tokenize(text):
    return re.findall(r"\w+|[^\w\s]",text,re.UNICODE)

def get_tokenized_wine_reviews(wine_reviews):
    # tokenize each wine review in the list and return
    return [custom_tokenize(text) for text in wine_reviews]

def flatten_tokenized_wine_reviews(tokenized_wine_reviews):
    return [token for tokenized_text in tokenized_wine_reviews for token in tokenized_text]

def get_wine_review_word_to_id(wine_review_tokens):
    wine_review_token_counter = Counter(wine_review_tokens)
    wine_review_vocab_words = wine_review_token_counter.most_common(VOCAB_SIZE)
    wr_word_to_id = { word:idx for idx, (word,_) in enumerate(wine_review_vocab_words)}
    wr_word_to_id[UNKOWN_WORD] = len(wr_word_to_id)
    wr_word_to_id[PAD_TOKEN] = len(wr_word_to_id)
    return wr_word_to_id

def get_wine_review_id_to_word(wr_word_to_id):
    return {id:word for word, id in wr_word_to_id.items()}


def enhanced_tokenize(text, word_to_id):
    tokens = custom_tokenize(text)
    return [token if token in word_to_id else UNKOWN_WORD for token in tokens]

def tokenize_and_convert_to_ids(text, word_to_id):
    tokens = enhanced_tokenize(text,word_to_id)
    return [word_to_id[token] for token in tokens]


def batch_tokenize(example,wr_word_to_id):
    """
    example will represent one element from Dataset batch
    """
    input_ids = tokenize_and_convert_to_ids(example["text"],wr_word_to_id)
    if len(input_ids) > MAX_LEN+1:
        return {"input_ids":input_ids[:MAX_LEN+1]}
    else:
        input_ids = input_ids + [wr_word_to_id[PAD_TOKEN]]*(MAX_LEN+1-len(input_ids))
        # for idx in range(len(input_ids),MAX_LEN+1):
        #     input_ids.append(wr_word_to_id[PAD_TOKEN])
        return {"input_ids":input_ids}

def get_input_ids_as_tensors(input_ids):
    #input_ids is a list of lists, where each sublist is a list of token IDs
    # we are building tensor with (N,L) where N will match dataset size and L is seq length
    return torch.cat([torch.tensor(input_ids_record).unsqueeze(0) for input_ids_record in input_ids],dim=0)

def get_x_and_y_from_input_ids_tensor(input_ids_tensor,seq_len=80):    
    return input_ids_tensor[:,:seq_len],input_ids_tensor[:,1:]




class WineReviewDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class TokenPositionEmbedding(nn.Module):
    def __init__(self,vocab_size, max_len,embed_dim,device=torch.device("cpu")):
        super(TokenPositionEmbedding,self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.pos_embedding = nn.Embedding(self.max_len, self.embed_dim)
    
    def forward(self,x):
        # x is (N,L) tensor where each row is a list of token IDs
        seq_len = x.size(-1)
        if seq_len <=self.max_len:
            positions = torch.arange(seq_len)
        else:
            positions = torch.arange(self.max_len)
        positions = positions.to(self.device)
        positions_embeddings = self.pos_embedding(positions)
        return self.embedding(x) + positions_embeddings


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim,droupout_rate=0.1, device = torch.device("cpu")) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.device = device
        self.droupout_rate = droupout_rate
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim,self.num_heads,batch_first=True, dropout=self.droupout_rate)
        self.ln1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn1 = nn.Linear(self.embed_dim,self.ff_dim)
        self.ffn2 = nn.Linear(self.ff_dim, self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim,eps=1e-6)
    
    def forward(self,inputs):
        # inputs is embedded batch of input_ids so it will have (N,L,E) shape
        batch_size, seq_len, embed_dim = inputs.size()        

        # we can use this as input to attn_mask argument of MultiHeadAttention when calling it
        # as you can see we are first using torch.tril to generate square matrix where values above diagonal are zero
        # then we set elements where values are zero to True, indicating those are the positions that we want to mask
        mask = torch.tril(torch.ones(seq_len, seq_len))==0
        mask = mask.to(self.device)

        attn_out, attn_weights = self.multihead_attn(inputs,inputs,inputs,attn_mask=mask)
        out1 = self.ln1(inputs + attn_out)
        ffn1 = self.ffn1(out1)
        ffn1 = F.relu(ffn1)
        ffn2 = self.ffn2(ffn1)
        ffn_out = F.dropout(ffn2,self.droupout_rate)
        return self.ln2(out1 + ffn_out), attn_weights


class GPT(nn.Module):
    def __init__(self, token_pos_embedding:TokenPositionEmbedding, transformer_block:TransformerBlock,embed_dim, vocab_size) -> None:
        super().__init__()        
        self.token_pos_embedding = token_pos_embedding
        self.transformer_block = transformer_block
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
    
    def forward(self,input_ids):
        embeddings = self.token_pos_embedding(input_ids)
        out,attn_weights = self.transformer_block(embeddings)
        # this will return (N, L , V) tensor which specified predicted tokens for every position
        # Lth element specifies probability of which word from Vocabulary to come next after L tokens in input_ids        
        scores = F.relu(self.linear(out))
        return scores,attn_weights

def softmax_over_gpt_scores(gpt_scores):
    return F.softmax(gpt_scores,dim=2)


class TextGenerator:
    def __init__(self,word_to_id, id_to_word,gpt_model,device=torch.device("cpu")) -> None:
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.gpt_model = gpt_model
        self.device = device
    
    def sample_from(self,probs,temperature):
        probs = probs ** (1/temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs),p=probs), probs
    
    def generate(self,start_prompt,max_tokens,temperature):
        start_tokens = tokenize_and_convert_to_ids(start_prompt,self.word_to_id)
        sample_token = None
        info = []

        while len(start_tokens) < max_tokens and sample_token != self.word_to_id[PAD_TOKEN]:
            x = np.array([start_tokens])
            x_tensor = torch.tensor(x)
            x_tensor = x_tensor.to(self.device)
            with torch.no_grad():
                scores, attn_weights = self.gpt_model(x_tensor)
                y = softmax_over_gpt_scores(scores)
                y_np = y.cpu().numpy()
                sample_token, probs = self.sample_from(y_np[0][-1],temperature)
                info.append({"prompt":start_prompt,"word_probs":probs,"attns":attn_weights[0,-1,:]})
                start_tokens.append(sample_token)
                start_prompt = f"{start_prompt} {self.id_to_word[sample_token]}"
        print(f"generated text : {start_prompt}")
        return info