import torch
import torch.nn as nn
import zipfile
import numpy as np
import io
from vocab import Vocab
import os

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(args, vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    print(args)
    print(args.dev[5:11])
    if 'imdb' in args.dev:
        if os.path.exists("trained_embedding_imdb.npy"):
            print("it exists!")
            with open('trained_embedding_imdb.npy', 'rb') as f:
                matrix = np.load(f)
            return matrix
        else:
            print("imdb embedding not found")

    if 'sst' in args.dev:
        if os.path.exists("trained_embedding_sst.npy"):
            print("it exists!")
            with open('trained_embedding_sst.npy', 'rb') as f:
                matrix = np.load(f)
            return matrix
        else:
            print("imdb embedding not found")

    # if can't find trained embeddings use random embeddings
    embedding_matrix = random_embedding(vocab, emb_file, emb_size)
    print("could not find the preloaded embedding so using untrained random")


    # Code to load in crawl 2M embeddings

    #fin = io.open(emb_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #data = {}
    #for line in fin:
    #    tokens = line.rstrip().split(' ')
        # data[tokens[0]] = map(float, tokens[1:])
    #    data[tokens[0]] = tokens[1:]

    #embedding_matrix = np.zeros(shape=(len(vocab), emb_size), dtype=np.float64)

    #for i in range(0, len(vocab)):
    #    word = Vocab.id2word(vocab, i)
    #    try:
    #        word_vector = list(map(float, data[word]))
    #        embedding_matrix[i] = word_vector
    #    except:
    #        embedding_matrix[i] = np.random.uniform(-.04, .04, 300)

        # set padding and <unk> to zero also tried giving them values but zero worked best
    #    embedding_matrix[0] = np.random.uniform(-.0, .0, 300)
    #    embedding_matrix[1] = np.random.uniform(-.0, .0, 300)

    #with open('crawl-2M-imdb-0rand.npy', 'wb') as f:
    #    np.save(f, embedding_matrix)

    return embedding_matrix

def random_embedding(vocab, emb_file, emb_size):

    embedding_matrix = np.zeros(shape=(len(vocab), emb_size), dtype=np.float64)
    v = .08
    for i in range(0, len(vocab)):
        embedding_matrix[i] = np.random.uniform(-1 * v, v, 300)

    embedding_matrix[0] = np.random.uniform(-.0, .0, 300)
    embedding_matrix[1] = np.random.uniform(-.0, .0, 300)

    return embedding_matrix

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters(args,vocab, tag_size)
        self.init_model_parameters(args, tag_size)

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy(args, vocab, args.emb_file, args.emb_size)

    def define_model_parameters(self, args, vocab, tag_size):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = nn.Embedding(len(vocab), args.emb_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.hidden_layer1 = nn.Linear(300, 300, bias=False)
        self.hidden_layer2 = nn.Linear(300, 300, bias=False)
        self.hidden_layer3 = nn.Linear(300, tag_size, bias=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_model_parameters(self, args, tag_size):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        self.dropout = torch.nn.Dropout(0.3)
        torch.nn.init.xavier_uniform_(self.hidden_layer1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer2.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer3.weight)
        self.hidden_layer1.requires_grad_(True)
        self.hidden_layer2.requires_grad_(True)
        self.hidden_layer3.requires_grad_(True)
        torch.nn.init.xavier_uniform_(self.embedding.weight)


    def copy_embedding_from_numpy(self, args, vocab, emb_file, emb_size):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        embeddings = load_embedding(args, vocab, emb_file, emb_size)
        #embeddings = random_embedding(vocab, emb_file, emb_size)

        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        sum_embedding = torch.zeros(size=(x.size(dim=0), 300)).to(self.device)
        for j in range(0,x.size(dim=0)):
            for i in range(0,x.size(dim=1)):
                sum_embedding[j] += self.embedding(x[j][i])
                # word dropout didn't help so I comment it out
                # word drop out
                #rand_num = np.random.uniform(0, 1, 1)
                #if (rand_num < .3):
                #    continue
                #else:
                #    sum[j] += self.embedding(x[j][i])

        # averages the outputs from the embeddings
        avg_pool = torch.div(sum_embedding, x.size(dim=1)).float()

        # first layer
        x = self.hidden_layer1(avg_pool)
        x = self.dropout(x)
        x = torch.nn.ReLU()(x)
        # second layer
        x = self.hidden_layer2(x)
        x = self.dropout(x)
        x = torch.nn.ReLU()(x)
        # third layer
        x = self.hidden_layer3(x)
        return x


