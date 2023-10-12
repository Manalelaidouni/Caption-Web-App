import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from utils import *
from running_wandb import yaml_load_with_wandb


class Encoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.encoder_size = cfg.encoder_size
        backbone = torch.hub.load(cfg.torch_hub_dir, cfg.torch_hub_model, pretrained=True, verbose=False).to(device)
        layers = list(backbone.children())[:-2]
        self.network = nn.Sequential(*layers)        
        self.finetune()
            
    def finetune(self):
        for params in self.network.parameters():
            params.requires_grad = self.cfg.FINETUNE_ENCODER
      
    def forward(self, images):
        feature_map = self.network(images) #  [bs, depth, encoder_size, encoder_size]
        #feature_map = F.adaptive_avg_pool2d(feature_map, self.encoder_size)
        return feature_map.permute(0, 2, 3, 1) # shape: [bs, encoder_size, encoder_size, depth]


class Attention(nn.Module):
    """ Implementing Bahdanau et al.’s attention mechanism
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.encoder_dim, cfg.attention_dim)
        self.fc2 = nn.Linear(cfg.decoder_dim, cfg.attention_dim)
        self.fc3 = nn.Linear(cfg.attention_dim, 1)
        self.depth = cfg.depth
        self.hidden_dim = cfg.hidden_dim

    def forward(self, enc_features, prev_hidden):
        """
        prev_hidden for a single time step : [bs, decoder_dim/hidden_dim]
        encoder_features : [bs, num_pixels, depth/encoder_dim] 
        """       
        bs = enc_features.shape[0]
     
        encoder_features = self.fc1(enc_features) # [bs, num_pixels, attention_dim]
        prev_hidden = self.fc2(prev_hidden) # [bs, attention_dim]
        prev_hidden = prev_hidden.unsqueeze(1) # [bs, 1, attention_dim]
        
        # compute global alignment scores: 
        # how well does the sequence input at position t with encoder outputs
        align_scores = encoder_features + prev_hidden # [bs, num_pixels, attention_dim]
        align_scores = self.fc3(torch.tanh(align_scores)) # [bs, num_pixels, 1]
   
        # get the normalized alignment scores ∈ [0,1] using softmax
        atten_weights = F.softmax(align_scores, dim=1) 

        # a weighted sum of attention weights and encoder output
        # matrix product : [bs, num_pixels, encoder_dim]
        context_vector =   torch.sum((enc_features * atten_weights), dim=1) # [bs, encoder_dim]
        
        return context_vector, atten_weights.squeeze(2) #[bs, encoder_dim], [bs, num_pixels]



class DecoderStep(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()  
        vocab_size = len(vocab)    
        self.cfg = cfg
        self.f_beta = nn.Linear(cfg.decoder_dim, cfg.encoder_dim)
        # embed_dim + depth 
        #self.lstm = nn.LSTM(cfg.embed_dim , cfg.hidden_dim, cfg.num_layers, cfg.bidirectional, batch_first=cfg.batch_first)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, vocab_size)
        self.lstm_cell   = nn.LSTMCell(cfg.embed_dim + cfg.encoder_dim, cfg.decoder_dim) # , bias=True
        self.attention = Attention(cfg)
        
        
    def forward(self, features, embeddings, h, c, gate_idx):
        """
        features : [16, 49, 512] [bs, num_pixels, depth/encoder_dim]
        embeddings : [bs, embed_dim]
        h, c : [bs, depth/encoder_dim]
        """

        # runs attention and LSTM step
        context_vector, alpha = self.attention(features[:gate_idx], h[:gate_idx])

        gate = torch.sigmoid(self.f_beta(h[:gate_idx])) 
        context_vector = gate * context_vector

        # generate a new word with previous word and attention weighted encoding
        h, c = self.lstm_cell(torch.cat([embeddings[:gate_idx, :], context_vector], dim=1), #embeddings[:gate_idx, :] only batch size changes
            (h[:gate_idx], c[:gate_idx]))
        
        preds = self.fc(self.dropout(h))
        
        return preds, alpha, h, c 


    def predict_step(self, features, embeddings, h, c):
        """runs attention and LSTM step"""

        context_vector, alpha = self.attention(features, h)
        gate = torch.sigmoid(self.f_beta(h)) 
        context_vector = gate * context_vector

        # generate a new word with previous word and attention weighted encoding
        h, c = self.lstm_cell(torch.cat([embeddings, context_vector], dim=1), (h, c))
        preds = self.fc(self.dropout(h))
        preds = F.log_softmax(preds, dim=1)

        return preds, alpha, h, c 



class Decoder(nn.Module):
    def __init__(self, cfg, vocab, device):
        super().__init__()
        # changing path to glove to access it from app.py 
        #cfg.GLOVE_PATH = glove_path if glove_path is not None else cfg.GLOVE_PATH

        self.vocab_size = len(vocab) 
        self.cfg = cfg
        self.vocab = vocab
        self.device = device
        self.encoder_dim = cfg.encoder_dim
        self.decoder_dim = cfg.decoder_dim
        self.init_h = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.init_c = nn.Linear(cfg.encoder_dim, cfg.decoder_dim)
        self.embed = nn.Embedding(self.vocab_size, cfg.embed_dim)
        self.decoder_step = DecoderStep(cfg, vocab)
        
        if cfg.use_glove_embeddings:
            #embedding_matrix= self._get_glove_matrix(cfg, vocab)
            #torch.save(embedding_matrix, 'embedding_matrix.pt')
            embedding_matrix = torch.load('./embedding_matrix.pt')
            self.embed.weight = nn.Parameter(embedding_matrix)

            #.from_pretrained(self.embedding_matrix, freeze=True, padding_idx=vocab[cfg.PAD_TOKEN])
            self.embed.padding_idx = vocab[cfg.PAD_TOKEN]
            
            if cfg.finetune_embedding:
                self.finetune_embeddings()
            else:
                self.finetune_embeddings(fine_tune=False)
     
        
    def finetune_embeddings(self, fine_tune=True):
        for p in self.embed.parameters():
            p.requires_grad = fine_tune


    def init_hidden_state(self, encoder_out):
        """encoder_out : (bs, num_pixels, encoder_dim=depth) (16, 49, 512)
        """

        # get the mean of the encoded image's dim=1
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)   

        return h, c #[16, 512] [bs, encoder_dim]


    def forward(self, features, targets, cap_lengths, teacher_forcing=1, random_top_k=0):
        """
        features : [16, 7, 7, 512] [bs, h1, w1, depth]
        targets : [16, 17] [bs, seq_len]
        cap_lenghts : [bs]
        """

        bs = features.shape[0]
        features = features.reshape(bs, -1, self.encoder_dim) # [bs, h x w, enc_dim==dec_dim==hidden_dim]
        
        num_pixels = features.shape[1] 

        h,c = self.init_hidden_state(features)

        embeddings = self.embed(targets) # [bs, seq_len, embed_dim]

        original_lengths = (cap_lengths).tolist()

        predictions = torch.zeros(bs, max(original_lengths), self.vocab_size).to(self.device)
        alphas = torch.zeros(bs, max(original_lengths) , num_pixels).to(self.device)
     
        decoder_input = torch.LongTensor([self.vocab[self.cfg.START_TOKEN]]).repeat(bs).to(self.device)
        decoder_input =  self.embed(decoder_input) # [bs, embed_dim]

        for step in range(max(original_lengths) ):
            gate_idx = sum([l>step for l in original_lengths])
            # feeds <sos> to decoder in first step 
            preds, alpha, h, c = self.decoder_step(features, decoder_input, h, c, gate_idx) # preds [bs, vocab_size] # alpha [bs, num_pixels=49]
            
            predictions[:gate_idx, step, :] = preds
            alphas[:gate_idx, step, :] = alpha

            # Use Scheduled Sampling
            use_teacher_forcing = random.random() < teacher_forcing
            
            if use_teacher_forcing:
                decoder_input = embeddings[:, step, :] # [bs, embed_dim] add step+1
           
            else:
                # when not using teacher forcing, either select the topmost choice word as the input to the next decoder step 
                # or randomly sample from a top k words 
                preds = F.log_softmax(preds, dim=-1)
                decoder_input = preds.argmax(dim=-1) # [bs]

                if random_top_k > 1:
                    indices, _ = preds.topk(random_top_k, dim=-1)# [bs, random_top_k]
                    decoder_input = torch.multinomial(indices, 1)
                    decoder_input = decoder_input.squeeze(1)  # [bs]

                decoder_input = self.embed(decoder_input) # [bs, embed_dim]

        predictions = F.log_softmax(predictions, dim=-1)  

        return predictions, original_lengths, alphas



    def predict_beam_search(self, encoder_output, beam_width, max_seq_len, processor):
        bs = encoder_output.shape[0] # [1, 7, 7, 512]
        encoder_output = encoder_output.view(bs, -1, self.encoder_dim)
        h, c = self.init_hidden_state(encoder_output)
        seq_list = [ [[], 0.0] for _ in range(beam_width)]
        decoder_input = torch.LongTensor([self.vocab[self.cfg.START_TOKEN]]).repeat(bs).to(self.device)
        decoder_input = self.embed(decoder_input) # [bs, embed_dim]

        with torch.no_grad():
            for step in range(1, max_seq_len+1):
                if step == 1:
                    prediction, alpha, h, c = self.decoder_step.predict_step(encoder_output, decoder_input, h, c) # prediction [bs, vocab_size]
                    prediction = F.log_softmax(prediction, dim=1)
                    top_words, top_scores = get_top_k(prediction, beam_width, processor)

                    for li, word, score in zip(seq_list, top_words, top_scores):
                        li[0].append(word)
                        li[1] += score

                    #print('first step done ', seq_list)
                    # [ [['returning'], 0.4039207994937897], [['peanut'], 0.3556343913078308] ]

                else:
                    new_seq_list = []
                    for idx in range(len(seq_list)):
                        # for every sequence in sequence list feed the last word in sequence to decoder
                        last_word = seq_list[idx][0][-1]
                        if last_word not in self.vocab.keys():
                            last_word = self.cfg.UNK_TOKEN
                            print('Unkown word')

                        decoder_input = torch.LongTensor([self.vocab[last_word]]).repeat(bs).to(self.device)
                        decoder_input = self.embed(decoder_input)

                        prediction, alpha, h, c = self.decoder_step.predict_step(encoder_output, decoder_input, h, c)
                        prediction = F.log_softmax(prediction, dim=1)

                        top_words, top_scores = get_top_k(prediction, beam_width, processor)

                        # add all combinations to list
                        for li, word, score in zip(seq_list, top_words, top_scores):
                            li[0].append(word)
                            li[1] += score
                            new_seq_list.append(seq_list)
                        
                        # generated number of sequences is K x K now
                        # pick top k sequences from the generated ones based on score
                        new_seq_list = pick_top_seq(new_seq_list, beam_width)


        # return a list of K sequences with max_seq_len along with their scores 
        seq_list.sort(key=lambda x: x[1], reverse=True)
        best_seq = seq_list[0]
        return best_seq


    def load_pre_trained_embedding(self):
        self.embed = self.embed.from_pretrained(self.embedding_matrix, freeze=True)
        self.embed.weight.requires_grad = self.cfg.finetune_embed


    @staticmethod
    def _get_glove_matrix(cfg, vocab):
        ''' Initialize embedding matrix of size (vocab_size, embed_dim) with vocab tokens mapped to Glove weights.
        1. construct a matrix of weights from glove for every vocab token.
        2. Load the glove matrix to nn.embedding
        3. choose to update weights in embedding matrix or freeze them.
        '''
        
        def load_embed_vectors(file_path, vocab_size, as_tensor=True):
            word2vector = {}
            with open(file_path, encoding="utf8") as f : 
                data = f.readlines()
            for datum in data:
                word = datum.strip().split()[0]
                embedding = list(map(float, datum.strip().split()[1:]))
            
                if as_tensor:
                    embedding = torch.FloatTensor(list(map(float, datum.strip().split()[1:])))

                word2vector[word] = embedding
            return word2vector

        word2vector = load_embed_vectors(cfg.GLOVE_PATH , len(vocab))

        embedding_matrix = torch.zeros(len(vocab), cfg.embed_dim)

        for idx, key in enumerate(vocab.keys()):
            if key in word2vector.keys():
                embedding_matrix[idx] = word2vector[key]

            else:
                embedding_matrix[idx] = torch.randn(cfg.embed_dim)

        return embedding_matrix #[vocab_size, embed_dim]



class EncoderDecoderWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, data, target, lengths):
        features = self.encoder(data)
        prediction, original_lengths, alphas = self.decoder(features, target, lengths)
        packed_prediction = pack_padded_sequence(prediction, original_lengths, batch_first=True).data # (data, batch_sizes, sorted_indices=None)
        packed_target = pack_padded_sequence(target, original_lengths, batch_first=True).data
        return prediction, original_lengths, alphas


