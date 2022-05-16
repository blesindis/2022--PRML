import os
import torch 
import torch.nn as nn
import numpy as np


VOCAB_FILENAME = 'vocab.txt'
STATIC_HYPER_FILENAME = 'static_hyper.json'
STATIC_EMBED_FILENAME = 'static.txt'


class Embedding(nn.Module):
    def __init__(self, vocab, model_name, dropout=0, word_dropout=0, word_type='en', requires_grad: bool=True, **kwargs):        
        super(Embedding, self).__init__()
        print("Pretraining {} embedding...".format(word_type))
        self.word_type = word_type
        if model_name:
            model_path = os.path.join('./model/words', model_name)
            print("Loading embedding from {}...".format(model_path))
            embedding = self._load_with_vocab(model_path, vocab)
        else:
            embedding = self._randomly_init_embed(len(vocab), embedding_dim=300)
            self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.index('<pad>'),
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.dropout_layer = nn.Dropout(dropout)
        self.word_dropout = word_dropout
        self._word_pad_index = vocab.index('<pad>')
        self._word_unk_index = vocab.index('<unk>')
        self.kwargs = kwargs
        self.vocab = vocab


    def forward(self, words):
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words

    
    def _load_with_vocab(self, embed_filepath, vocab, padding='<pad>', unknown='<unk>'):
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            #Check the first line
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            #Calculate
            matrix = {}
            matrix[vocab.index('<pad>')] = torch.zeros(dim)
            matrix[vocab.index('<unk>')] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                parts = line.strip().split()                
                word = ''.join(parts[:-dim])
                nums = parts[-dim:]
                if word in vocab:
                    index = vocab.index(word)
                    matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=np.float32, count=dim))
                    found_count += 1
                if word == unknown:
                    found_unknown = True
            print("Found {} out of {} words in the pre-training embedding of vocab '{}'.".format(found_count, len(vocab), self.word_type))
            #Create entry for unknown
            for i in range(len(vocab)):
                if i not in matrix:
                    if found_unknown:
                        matrix[i] = matrix[vocab.index('<unk>')]
                    else:
                        matrix[index] = None
            #Establish Embedding
            vectors = self._randomly_init_embed(len(matrix), dim)
            unknown_idx = vocab.index('<unk>')
            self.register_buffer('words_to_words', torch.full((len(vocab), ), fill_value=unknown_idx, dtype=torch.long).long())
            index = 0
            for i in range(len(vocab)):
                if i in matrix:
                    vec = matrix.get(i)
                    if vec is not None:
                        vectors[index] = vec
                    self.words_to_words[i] = index
                    index += 1

            return vectors


    def _randomly_init_embed(self, num_embeddnig, embedding_dim):
        embed = torch.zeros(num_embeddnig, embedding_dim)
        nn.init.uniform_(embed, -np.sqrt(3/embedding_dim), np.sqrt(3/embedding_dim))
        return embed


    def drop_word(self, words):
        if self.word_dropout > 0 and self.training:
            mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float)
            mask = torch.bernoulli(mask).eq(1)
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words


    def dropout(self, words):
        return self.dropout_layer(words)


    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, STATIC_EMBED_FILENAME), 'w', encoding='utf-8') as f:
            f.write('{} {}\n'.format(self.embedding.num_embeddings, self.embedding.embedding_dim))
            word_count = 0
            saved_word = {}
            valid_word_count = 0
            for i in range(len(self.words_to_words)):
                word = self.vocab