import os
import torch
import nltk
import random
import time
#import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate import bleu_score 
from nltk.translate.bleu_score import SmoothingFunction
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tag import pos_tag
from PIL import Image

try :
    from torch_lr_finder import LRFinder, TrainDataLoaderIter
except ImportError:
    LRFinder = None
    TrainDataLoaderIter = None


class EarlyStopping:
    def __init__(self, patience, mode='min'):
        assert mode in {'min', 'max'}, "mode could either be 'min' or 'max'"
        self.patience = patience
        self.mode = mode
        self.count = 0
        self.prev_score = None
        self.stop = False

    def __call__(self, curr_score, epoch):
        if self.prev_score is None:
            pass

        elif self._is_score_improved(curr_score, self.prev_score):
            self.count = 0
        
        elif not self._is_score_improved(curr_score, self.prev_score):
            self.count +=1
            if self.count == self.patience:
                self.stop = True

        self.prev_score = curr_score

    def _is_score_improved(self, curr_score, prev_score):
        if self.mode == 'min':
            return prev_score > curr_score

        elif self.mode == 'max':
            return prev_score < curr_score


if LRFinder and TrainDataLoaderIter:
    class LRFinderWrapper(LRFinder):
        def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
            self.model.train()
            total_loss = None  # for late initialization
            self.optimizer.zero_grad()
            for i in range(accumulation_steps):
                inputs, labels, lengths = next(train_iter)
                inputs, labels = self._move_to_device(
                    inputs, labels, non_blocking=non_blocking_transfer)

                # Forward pass
                prediction, original_lengths, _= self.model(inputs, labels, lengths)
                packed_prediction = pack_padded_sequence(prediction, original_lengths, batch_first=True).data
                packed_labels = pack_padded_sequence(labels, original_lengths, batch_first=True).data

                loss = self.criterion(packed_prediction, packed_labels)
                # Loss should be averaged in each step
                loss /= accumulation_steps
                # Backward pass
                loss.backward()
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            self.optimizer.step()

            return total_loss.item()


        def validate(self, val_iter, non_blocking_transfer=True):
            # Set model to evaluation mode and disable gradient computation
            running_loss = 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels, lengths in val_iter:
                    # Move data to the correct device
                    inputs, labels = self._move_to_device(
                        inputs, labels, non_blocking=non_blocking_transfer)

                    # Forward pass and loss computation
                    outputs = self.model(inputs, lengths)

                    prediction, original_lengths, _= self.model(inputs, lengths)
                    packed_prediction = pack_padded_sequence(prediction, original_lengths, batch_first=True).data
                    packed_labels = pack_padded_sequence(labels, original_lengths, batch_first=True).data

                    loss = self.criterion(packed_prediction, packed_labels)
                                
                    running_loss += loss.item() * len(labels)

            return running_loss / len(val_iter.dataset)



    class CustomTrainIter(TrainDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data): 
            inputs, labels, lengths = batch_data 
    
            return inputs, labels, lengths
        
        def __next__(self):
            try:
                batch = next(self._iterator)
                inputs, labels, lengths  = self.inputs_labels_from_batch(batch)
            except StopIteration:
                if not self.auto_reset:
                    raise
                self._iterator = iter(self.data_loader)
                batch = next(self._iterator)
                inputs, labels, lengths  = self.inputs_labels_from_batch(batch)

            return inputs, labels, lengths 

    

def train_all(cfg, encoder, decoder, device, train_loader, val_loader, enc_optimizer, dec_optimizer, criterion, start_epoch, processor, track_metric, encoder_scheduler=None, decoder_scheduler=None, early_stop=None, log_to_wandb=True):
    if track_metric == 'bleu':
        mode = 'max'
        best_score = -float('inf')
    elif track_metric == 'loss':  
        mode = 'min'
        best_score = float('inf') 
    else:
        raise ValueError(f'Metric {track_metric} not implemented. Select either  `bleu ` or  `loss`')

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        start = time.time()
        train_loss, train_bleu = train(encoder, decoder, device, train_loader, enc_optimizer, dec_optimizer, criterion, epoch, cfg, processor)
        val_loss, val_bleu = validate(encoder, decoder, device, criterion, val_loader, processor )
        # save scores for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if log_to_wandb:
            wandb.log({"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss})
            wandb.log({"epoch": epoch, "val_bleu": val_bleu, "train_bleu": train_bleu})

        # save the best loss result so far
        score_to_track = val_loss if track_metric=='loss' else val_bleu
        if mode == 'min':
            print(f"Epoch: {epoch}/{cfg.epochs} - Training loss: {train_loss:.4f} - Validation loss: {val_loss:.4f}")
            is_best = score_to_track < best_score 
            op = min

        elif mode == 'max':
            print(f"Epoch: {epoch}/{cfg.epochs} - Training Bleu {train_bleu:.4f} - Validation Bleu: {val_bleu:.4f}")
            is_best = score_to_track > best_score 
            op = max

        if is_best:
            save_model(epoch, encoder, decoder, enc_optimizer, dec_optimizer, cfg)
            best_score = op(score_to_track, best_score)
            if mode == 'min':
                print(f'Saved model with lowest validation score: {best_score:.4f}')
            elif mode == 'max':
                print(f'Saved model with highest validation score: {best_score:.4f}')

        if isinstance(early_stop, EarlyStopping):
            # updating using loss function
            early_stop(score_to_track , epoch)    
            if early_stop.stop:
                print(f"Validation score did not improve for {cfg.patience} epochs, Early stopping now ..")
                break
      
        if cfg.use_scheduler:
            if cfg.scheduler == 'reduce_on_plateau':
                if cfg.FINETUNE_ENCODER:
                    encoder_scheduler.step(score_to_track)
                decoder_scheduler.step(score_to_track)
            else:
                if cfg.FINETUNE_ENCODER:
                    encoder_scheduler.step()
                decoder_scheduler.step()
        print(f'Time it took to train one epoch is: {time.time()-start:.2f} sec ')

    print('Finished Training')
    return train_losses, val_losses



def save_model(epoch, encoder, decoder, enc_optimizer, dec_optimizer, cfg, save_to_wandb=False):
    if not os.path.exists(cfg.CHECKPOINT_PATH):
        os.makedirs(cfg.CHECKPOINT_PATH)

    checkpoint_fname = f'checkpoint_at_epoch_{epoch}'
    path = os.path.join(cfg.CHECKPOINT_PATH, checkpoint_fname)

    checkpoint = {'epoch_num' : epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'enc_optimizer': enc_optimizer.state_dict() if enc_optimizer is not None else None,
        'dec_optimizer': dec_optimizer.state_dict() }
   
    torch.save(checkpoint, path)

    if save_to_wandb:
        torch.save(checkpoint, os.path.join(wandb.run.dir, checkpoint_fname))
        wandb.save(path, base_path=cfg.CHECKPOINT_PATH)
    

def load_checkpoint(cfg, encoder, decoder, enc_optimizer=None, dec_optimizer=None, resume_training=False, checkpoint_path=None):
    # checkpoint_path arg specified for ec2 instance path
    checkpoint_path = checkpoint_path if checkpoint_path is not None else cfg.CHECKPOINT_PATH 

    if os.path.exists(checkpoint_path):
        path = os.path.join(checkpoint_path, cfg.checkpoint_fname)
        print(f'Loading weights ...')
        state = torch.load(path, map_location='cpu') 
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])
        
        if resume_training and dec_optimizer is not None:
            dec_optimizer.load_state_dict(state['dec_optimizer'])    
            if cfg.FINETUNE_ENCODER:    
                enc_optimizer.load_state_dict(state['enc_optimizer'])   
            epoch_num = state['epoch_num']
            print('Loading weights to resume training is done.')
        else : 
            epoch_num = None
    else:
        print('File is not being recognized')
        raise FileNotFoundError('Path to checkpoint is not valid')
        # TODO: load checkpoint form wandb using wandb API
    
    return  encoder, decoder, enc_optimizer, dec_optimizer, epoch_num



def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train(encoder, decoder, device, train_loader, enc_optimizer, dec_optimizer, criterion, epoch, config, processor, save_on_exit=True):
    encoder.train()
    decoder.train()
    sum_losses = 0
    avg_loss = 0 
    all_bleu = []

    for batch_idx, (data, target, lengths) in enumerate(train_loader):
        try:
            data, target, lengths = data.to(device), target.to(device), lengths.to(device)
            features = encoder(data).to(device)
            prediction, original_lengths, alphas = decoder(features, target, lengths)
            """
            pred = prediction.argmax(dim=-1) #[bs, max_seq_len]
            captions = [from_idx_to_token(p, processor, to_sentence=True) for p in pred] 
            targets = [from_idx_to_token(t, processor, to_sentence=True) for t in target] 

            print('Target :',*targets, sep='\n')
            print('Pred: ', *captions, sep='\n')
            """
            # to remove all padding before passing to loss function        
            packed_pred = pack_padded_sequence(prediction, original_lengths, batch_first=True).data # (data, batch_sizes, sorted_indices=None)
            # packed_pred [num_non_padded_items, vocab_size] 
            packed_target = pack_padded_sequence(target, original_lengths, batch_first=True).data

            loss = criterion(packed_pred, packed_target)  

            if enc_optimizer is not None:
                enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            
            loss.backward()
            
            # gradient clipping
            if config.clip > 0:
                if enc_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)
                
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)

            if enc_optimizer is not None:
                enc_optimizer.step()
            dec_optimizer.step()    

            # compute loss
            sum_losses += loss.item()
            avg_loss = sum_losses/(batch_idx+1)

            # compute bleu score 
            pred = prediction.argmax(dim=-1)
            bleu = bleu_score_for_batch(pred, target, processor)
            all_bleu.append(bleu)

        except KeyboardInterrupt:
            print('Exiting program ..')
            if save_on_exit:
                save_model(epoch, encoder, decoder, enc_optimizer, dec_optimizer, config)
                break

    avg_bleu = np.mean(all_bleu)
    return sum_losses/len(train_loader.dataset) , avg_bleu



def validate(encoder, decoder, device, criterion, test_loader,  processor):
    encoder.eval()
    decoder.eval()
    
    sum_losses = 0    
    all_bleu = []
    
    with torch.no_grad():
        for batch_idx, (data, target, lengths) in enumerate(test_loader):

            data, target, lengths = data.to(device), target.to(device), lengths.to(device)

            features = encoder(data).to(device)
            prediction, original_lengths, alphas = decoder(features, target, lengths, teacher_forcing=0.5) 
      
            # compute loss
            packed_prediction = pack_padded_sequence(prediction, original_lengths, batch_first=True).data # (data, batch_sizes, sorted_indices=None)
            packed_target = pack_padded_sequence(target, original_lengths, batch_first=True).data
            
            loss = criterion(packed_prediction, packed_target)  
            sum_losses += loss.item() 

            # compute bleu score
            pred = prediction.argmax(dim=-1) #[bs, max_seq_len]
            bleu = bleu_score_for_batch(pred, target, processor)
            all_bleu.append(bleu)
            
            captions = [from_idx_to_token(p, processor, to_sentence=True) for p in pred] 
    
            targets = [from_idx_to_token(t, processor, to_sentence=True) for t in target] 
         
            print('Target :',*targets, sep='\n')
            print('Pred: ', *captions, sep='\n')
    
    avg_bleu = np.mean(all_bleu)
    avg_loss = sum_losses/len(test_loader.dataset)
    return avg_loss, avg_bleu


 
def inference_image(encoder, decoder, device, image, processor, beam_width, max_seq_len):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.to(device)
        features = encoder(image).to(device)
        prediction = decoder.predict_beam_search(features, beam_width, max_seq_len, processor)
        return prediction[0]



def bleu_score_for_batch(predictions, targets, processor):
    all_preds = []
    all_targets = []
    for pred in predictions:
        pred = pred[pred!=0]   # remove padding
        all_preds.append(from_idx_to_token(pred, processor)) 

    for target in targets: 
        target = target[target!=0] #[1:-1] # remove pad, end and start tokens
        all_targets.append([from_idx_to_token(target, processor)])
    
    #  added smoothing function so if there is no overlap, the result would not be a harsh 0.
    bleu = bleu_score.corpus_bleu(all_targets, all_preds, smoothing_function=SmoothingFunction().method1) # weights=(0.25, 0.25, 0.25, 0.25) 
    return bleu



def remove_transform(image):
    image = image.squeeze(0)
    inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255])
    # unormalize
    img = inv_normalize(image)

    img = img.permute(1,2,0)
    # convert to numpy
    img = img.detach().cpu().numpy()
    return img #(img * 255).astype(np.uint8)




def inference_batch(encoder, decoder, device, test_batch, processor):
    encoder.eval()
    decoder.eval()
    captions = []
    targets = []
    
    with torch.no_grad():
        data, target, lengths = test_batch
        data, target, lengths = data.to(device), target.to(device), lengths.to(device)
        features = encoder(data).to(device)
        prediction, original_lengths, alphas = decoder(features, target, lengths, teacher_forcing=0)
        pred = prediction.argmax(dim=-1)
        bleu = bleu_score_for_batch(pred, target, processor)
        captions = [from_idx_to_token(p, processor, to_sentence=True) for p in pred]
        targets = [from_idx_to_token(t, processor, to_sentence=True) for t in target] 
        
    return captions, targets, bleu



def from_idx_to_token(cap, processor,to_sentence=False):
    c = []
    try:
        for idx in cap:
            c.append(processor.idx2token[idx.item()])
    except KeyError:
        pass

    if to_sentence:
        c = ' '.join(c)
    return c



def plot_losses(train_scores, test_scores, img_name):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    plt.plot(train_scores, color='red', label = 'Train Loss')
    plt.plot(test_scores, color='blue', label = 'Validate Loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss value')
    plt.legend(loc='upper right')
    plt.title("Train vs Validation Loss", fontstyle='italic')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(curr_dir, img_name))
    #wandb.log({'training vs validation score': wandb.Image(os.path.join(curr_dir, img_name))})
    plt.show()



def pick_top_seq(list_seq, beam_width):
    list_seq.sort(key=lambda x: x[1], reverse=True)
    return list_seq[:beam_width]


def get_top_k(predictions, k, processor):
    top_scores, top_indices  = torch.topk(predictions.squeeze(0), k, dim=-1)
    top_words = [processor.idx2token[idx.item()]for idx in top_indices]
    return top_words, top_scores.cpu().detach().numpy()


def remove_padding(sequences, original_lenghts):
    new_seqs = []
    for seq, length in zip(sequences, original_lenghts):
        # remove <sos> and <eos>
        orginal = seq[1: length-1]
        new_seqs.append(orginal)
    return new_seqs


def unpadd_all(targets, prediction, original_lenghts):
    # 1. remove padding from both tensors along with special tokens
    unpadded_target = remove_padding(targets, original_lenghts)
    unpadded_pred = remove_padding(prediction, original_lenghts)
    return torch.as_tensor(unpadded_target), torch.as_tensor(unpadded_pred)



def transform_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform =  transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))]) 
    image = transform(image)
    return image



def inference_beam_search(encoder, decoder, idx2token, vocab, cfg, device, data_loader=None, image_path=None, beam_size=10):
    if image_path is not None:
        image = transform_image(image_path)

    elif data_loader is not None:
        imgs, _, _ = next(iter(data_loader))
        index = random.randint(0, cfg.batch_size-1)
        image = imgs[index]

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device) # [1, 3, 224, 224]
        features = encoder(image).to(device) # [1, 7, 7, 2048])
        # features post processing
        features = features.reshape(1, -1,  decoder.encoder_dim) 
        features = features.expand(beam_size, features.shape[1], decoder.encoder_dim) 

        previous_tokens = torch.LongTensor([[decoder.vocab[cfg.START_TOKEN]]] * beam_size).to(device)
        top_probs = torch.zeros(beam_size, 1).to(device)
        all_seqs = previous_tokens

        h, c = decoder.init_hidden_state(features)
        accumulated_scores = torch.tensor(1).expand(10)
        step = 1

        while True:
            embeddings = decoder.embed(previous_tokens).squeeze(1)
            outputs, _, h, c = decoder.decoder_step.predict_step(features, embeddings, h, c)

            if step == 1:
                next_tokens = torch.LongTensor([12, 12, 517, 12,  18, 93, 432, 95, 104, 128])
                top_probs = torch.tensor(1).expand(10) 

            else:
                top_probs, top_words = outputs[0].topk(beam_size, dim=0)
                next_tokens = top_words


            all_seqs = torch.cat([all_seqs, next_tokens.unsqueeze(1)], dim=1)

            accumulated_scores = accumulated_scores * top_probs
            with_scores =  [(seq, score) for seq, score in zip(all_seqs, accumulated_scores)]

            previous_tokens = next_tokens.unsqueeze(1)

            if step > 7:
                break
            step += 1

    # remove repeating words
    no_duplicates_list = []
    for idx, seq in enumerate(all_seqs) :
      l = []
      for token in seq:
        if token not in l:
          l.append(token)
        else:
          pass
      no_duplicates_list.append(l)

    final_caps = []
    for seq in no_duplicates_list:
        cap = [idx2token[idx.item()] for idx in seq if idx.item() in idx2token.keys() and idx !=vocab[cfg.END_TOKEN]]
        word_tag = pos_tag(cap[1:])
        if  word_tag[-1][-1] in ['VBP', 'IN', 'VBZ','CC']:
            del cap[-1]

        caption = ' '.join(cap + [cfg.END_TOKEN])
        final_caps.append(caption)

    return final_caps[0]



def my_beam_search(encoder, decoder, idx2token, vocab, cfg, device, data_loader=None, image_path=None, k_beam=3):
    if image_path is not None:
        image = transform_image(image_path)

    elif data_loader is not None:
        imgs, _, _ = next(iter(data_loader))
        index = random.randint(0, cfg.batch_size-1)
        image = imgs[index]

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device) # [1, 3, 224, 224]
        features = encoder(image).to(device) # [1, 7, 7, 2048])
        features = features.reshape(1, -1,  decoder.encoder_dim)
        h, c = decoder.init_hidden_state(features)

        # pass first input
        first_input = torch.LongTensor([decoder.vocab[cfg.START_TOKEN]]).repeat(1).to(device)
        first_input = decoder.embed(first_input) # [1, 50]
        output, _, h, c = decoder.decoder_step.predict_step(features, first_input, h, c)
        top_probs, top_words = output.topk(k_beam, dim=1) # shape of top_words

        # change first step outputs
        top_words = torch.LongTensor([[12, 12, 517, 18, 93, 432, 95, 104, 128, 6]])
        top_probs = torch.LongTensor([[1]*k_beam])

        # first update of all_seqs
        all_seqs = [[[decoder.vocab[cfg.START_TOKEN]], 1]]* k_beam # [[[1], 0.0], [[1], 0.0], [[1], 0.0]]
        all_final = []
        updated_all_seqs = []
        for seq, word, prob in zip(all_seqs, top_words[0], top_probs[0]):
          new_seq = [seq[0] + [word.item()], seq[1] * prob.item()]
          updated_all_seqs.append(new_seq)

        top_words = top_words[0]
        step = 1

        while step <20:
          next_words = []
          next_probs = []

          for idx, word in enumerate(top_words):
            embed_input = decoder.embed(word.repeat(1).to(device))
            output, _, h, c = decoder.decoder_step.predict_step(features, embed_input, h, c)
            top_probs, top_words = output.topk(k_beam, dim=1)

            next_words.extend(top_words[0].tolist())
            next_probs.extend(top_probs[0].tolist())

          # create new_updated_seq with KxK sequence
          repeated_seqs = [item for item in updated_all_seqs for _ in range(k_beam)]

          all_seqs = []
          for seq, word, prob in zip(repeated_seqs, next_words, next_probs):
            new_seq = [seq[0] + [word], seq[1] * prob]
            all_seqs.append(new_seq)

          all_final.extend(all_seqs)
          # select top k seqs from all_seqs
          top_k_seqs = sorted(all_seqs, key=lambda x: x[1], reverse=True)[:k_beam]
          # get last words in each of top_k_seqs
          top_words = torch.tensor([seq[0][-1] for seq in top_k_seqs])
          step += 1
          updated_all_seqs = top_k_seqs

        return top_k_seqs
       

