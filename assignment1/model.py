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


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    if os.path.exists("crawl-2M-imdb-0rand.npy"):
        print("it exists!")
        with open('crawl-2M-imdb-0rand.npy', 'rb') as f:
            matrix = np.load(f)
        return matrix
    else:
        print("not found :(")
    fin = io.open(emb_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        # data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = tokens[1:]
    #embeddings = data

    embedding_matrix = np.zeros(shape=(len(vocab), emb_size), dtype=np.float64)
    num_bad_loads = 0
    for i in range(0, len(vocab)):
        word = Vocab.id2word(vocab, i)
        #if i<100:
        #    print(word)
        try:
            word_vector = list(map(float, data[word]))
            embedding_matrix[i] = word_vector
        except:
            embedding_matrix[i] = np.random.uniform(-.04, .04, 300)
            num_bad_loads += 1

        embedding_matrix[0] = np.random.uniform(-.0, .0, 300)
        embedding_matrix[1] = np.random.uniform(-.0, .0, 300)
    print(num_bad_loads)
    print(len(vocab))
    print(num_bad_loads/len(vocab))

    with open('crawl-2M-imdb-0rand.npy', 'wb') as f:
        np.save(f, embedding_matrix)

    return embedding_matrix

def random_embedding(vocab, emb_file, emb_size):

    embedding_matrix = np.zeros(shape=(len(vocab), emb_size), dtype=np.float64)
    v = 1
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
            self.copy_embedding_from_numpy(vocab, args.emb_file, args.emb_size)

    def define_model_parameters(self, args, vocab, tag_size):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        self.embedding = nn.Embedding(len(vocab), args.emb_size)
        #self.hidden_layer1 = nn.Linear(300,300)
        #self.hidden_layer2 = nn.Linear(300,300)
        #self.hidden_layer3 = nn.Linear(300,1)

        self.dropout = torch.nn.Dropout(0.3)
        self.hidden_layer1_test = nn.Linear(300, 300, bias=False)
        self.hidden_layer2_test = nn.Linear(300, 300, bias=False)
        self.hidden_layer3_test = nn.Linear(300, tag_size, bias=False)
        #self.hidden_layer2 = nn.Linear(300, 300)
        #self.hidden_layer3 = nn.Linear(300, 1)

        #self.layers = nn.Sequential(
        #    nn.Linear(300,300,nn.ReLU)
        #    nn.Linear(300,300)
        #
        #)

        #self.parameters()
        #raise NotImplementedError()

    def init_model_parameters(self, args, tag_size):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        v = .08
        dense_v = np.sqrt(6)/(np.sqrt(600))
        out_v = np.sqrt(6)/np.sqrt(300+tag_size)
        random_weights1 = np.random.uniform(-1 * dense_v, dense_v, (300, 300))
        random_weights2 = np.random.uniform(-1 * dense_v, dense_v, (300, 300))
        random_weights3 = np.random.uniform(-1 * out_v, out_v, (300, tag_size))
        #random_weights4 = np.random.uniform(-1 * v, v, (300, tag_size))
        #self.hidden_layer1 = torch.tensor(random_weights1)
        #self.hidden_layer2 = torch.tensor(random_weights2)
        #self.hidden_layer3 = torch.tensor(random_weights3)
        self.hidden_layer1 = nn.Parameter(torch.tensor(random_weights1).float())
        self.hidden_layer2 = nn.Parameter(torch.tensor(random_weights2).float())
        self.hidden_layer3 = nn.Parameter(torch.tensor(random_weights3).float())
        #self.hidden_layer4 = nn.Parameter(torch.tensor(random_weights4).float())
        #self.hidden_layer1.weight.copy_(random_weights1)
        #self.hidden_layer2.weight.copy_(random_weights2)
        #self.hidden_layer3.weight.copy_(random_weights3)
        self.dropout = torch.nn.Dropout(0.3)
        #raise NotImplementedError()
        torch.nn.init.xavier_uniform_(self.hidden_layer1_test.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer2_test.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer3_test.weight)
        self.hidden_layer1_test.requires_grad_(True)
        self.hidden_layer2_test.requires_grad_(True)
        self.hidden_layer3_test.requires_grad_(True)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.requires_grad_(False)



    def copy_embedding_from_numpy(self, vocab, emb_file, emb_size):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        embeddings = load_embedding(vocab, emb_file, emb_size)
        #embeddings = random_embedding(vocab, emb_file, emb_size)
        #print(embeddings)
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        #raise NotImplementedError()

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

        #input = torch.LongTensor([2])
        #print(self.embedding(input))
        sum = torch.zeros(size=(x.size(dim=0), 300)) #x.size(dim=0)
        for j in range(0,x.size(dim=0)):
            for i in range(0,x.size(dim=1)):
                #word drop out
                #rand_num = np.random.uniform(0, 1, 1)
                sum[j] += self.embedding(x[j][i])
                #print(type(sum[j]))
                #if (rand_num < .3):
                #    sum[j] += self.embedding(x[j][i])
                    #continue
                #else:
                #    sum[j] += self.embedding(x[j][i])

        avg_pool = torch.div(sum, x.size(dim=1)).float()
        #print(avg_pool)
        #print(avg_pool)
        #print(avg_pool.size())
        #print(self.hidden_layer1.size())
        #avg_pool = self.dropout(avg_pool)
        print("stuff")
        x = self.hidden_layer1_test(avg_pool)
        #x = torch.matmul(avg_pool, self.hidden_layer1)

        #x = self.dropout(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer2_test(x)
        #x = torch.matmul(x, self.hidden_layer2)

        #x = self.dropout(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer3_test(x)

        #x = torch.matmul(x, self.hidden_layer3)

        #x = torch.nn.ReLU()(feature_layer3)
        #x = self.dropout(x)
        #feature_layer4 = torch.matmul(x, self.hidden_layer4)


        return x
        #return output
        #raise NotImplementedError()

