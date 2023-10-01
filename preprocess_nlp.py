
import os
import string
from collections import Counter
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
# to not throw bad zip file error
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from running_wandb import yaml_load_with_wandb



class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        TOKENS_PATH = os.path.join(cfg.TEXT_PATH, 'Flickr8k.token.txt')
        self.img_to_captions = self._filename_to_captions(TOKENS_PATH)
        self._split_data_files(self.img_to_captions, cfg)
        self.vocab = self.get_vocab(cfg)
        self.idx2token = {idx:token for token, idx in self.vocab.items()}
        

    def _filename_to_captions(self, captions_path):
        ''' Creates a dictionary that maps each image filename to its list of captions.
        Args:
        captions_path (str): path to file with all filenames and their captions.
        Returns:
        fname_dict (dict): dictionary with each filename as key and list of 5 captions as value.
        '''
        fname_dict = {}
        data = self.load_text(captions_path)
        for line in data:
            # Skip empty lines
            if len(line) < 1:
                continue
            element = line.split("#")
            # Create a list of captions for each filename
            if element[0] in fname_dict:
                fname_dict[element[0]].append(element[1][2:])
            else:
                fname_dict[element[0]] = [element[1][2:]]
        return fname_dict


    def clean_text(self, text, cfg):
        """ Cleans text by normalizing, removing punctuations and finally tokenizing it.
        """
        if cfg.remove_punct:
            text = self._remove_punctuation(text)
        
        tokens = self._tokenize_text(text)

        if cfg.remove_stopwords:
            # Note: 'a' has the highest frequency in captions, 46776 times
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]

        if cfg.lemmatize:
            tokens =  self._lemmatize_tokens(tokens) 
        
        if cfg.stemmize:
            tokens =  self._stemmize_tokens(tokens) 

        if cfg.remove_numbers:
            tokens = [token for token in tokens if token.isalpha()]

        # removes only 'a' and one letter tokens
        tokens = [token for token in tokens if len(token)> 1]
        tokens = [cfg.START_TOKEN] +  tokens + [cfg.END_TOKEN]
        return tokens


    def _remove_punctuation(self, text):
        transtable = str.maketrans(' ', ' ', string.punctuation)
        text = text.translate(transtable).strip().lower()
        return text


    def _tokenize_text(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens
     

    def _lemmatize_tokens(self, tokens):
        """ Does morphological analysis of the words, e.g: better : good; siblings :sibling 
        """
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens] 
        return lemmas


    def _stemmize_tokens(self, tokens):
        """ Reduces words to root e.g, walking or walkers becomes walk.
        """
        stemmer = PorterStemmer()
        stems = [stemmer.stem(token) for token in tokens] 
        return stems


    def load_text(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                # skip last empty string
                if line == " ":
                    continue
                data.append(line.strip())
        return data


    def get_vocab(self, cfg, mode='train'):
        if mode not in {'train', 'dev', 'test'}:
            raise ValueError("Expected either `train`, `dev` or `test` as mode, but got `{mode}`")

        data_path = os.path.join(cfg.TEXT_PATH, f'{mode}_dataset.txt')
        data = self.load_text(data_path)

        count = Counter()
        print(f'Creating vocab from {mode} data ... ')
        for elem in data:
            if len(elem) < 1:
                continue
            _,  caption = elem.split('\t')
            tokens = self.clean_text(caption, cfg)
            count.update(tokens)

        vocab = {token:idx for idx, (token, cnt) in enumerate(count.items(), 2)}
        # cfg.START_TOKEN and cfg.START_TOKEN are already in vocab from loaded text
        vocab[cfg.PAD_TOKEN] = 0
        vocab[cfg.UNK_TOKEN] = 1

        return vocab


    def _split_data_files(self, dictionary, cfg):
        ''' Creates file with images filenames and their captions with the start and end tokens for a single data split.
        
        Args:
            mode (str) : specifies the data split.
            dictionary (dict) : maps each image filename to a list of its 5 captions.
                
        '''
        modes = {'train', 'dev', 'test'}
        for mode in modes:
            data_path = os.path.join(cfg.TEXT_PATH, f'Flickr_8k.{mode}Images.txt')
            
            save_path = os.path.join(cfg.TEXT_PATH, f'{mode}_dataset.txt')
            filenames = self.load_text(data_path)
            f = open(save_path, 'wb')
            
            for fname in filenames:
                if len(fname) < 1:
                    continue
                for cap in self.img_to_captions[fname]: 
                    f.write(bytes(fname+ '\t' + cap  + '\n', encoding='utf-8'))
            f.close()  



class Flicker8K(Dataset):
    def __init__(self, mode, cfg, processor, transform=None):
        super().__init__()
        assert mode in {'train', 'dev', 'test'}, f"""Invalid mode input.
        Expected either `train`, `dev` or `test`, but got `{mode}`."""
        self.mode = mode
        self.cfg = cfg
        self.transform = transform
        self.processor =  processor
        self.data_folder = cfg.TEXT_PATH 
        self.images_folder = cfg.IMG_PATH
        self.vocab = self.processor.vocab
        self.img_to_captions = self.processor.img_to_captions
        data_path = os.path.join(self.data_folder, f'{mode}_dataset.txt')
        if mode == 'train' :
            self.data_list = self.processor.load_text(data_path)   

        if mode in ['dev', 'test'] :
            self.data_list = self.processor.load_text(data_path)


    @property
    def data_path(self):
        return os.path.join(self.data_folder, f'{self.mode}_dataset.txt')

    def __getitem__(self, idx):
        line = self.data_list[idx]
        fname, caption = line.split('\t')

        # get image 
        image_path = os.path.join(self.images_folder, fname)
        # pretrained CNN expects RGB not BGR images
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            tensor_image =  self.transform(image)

        else:
            # if image augmentation is not applied
            # then just convert the PIL image to tensor
            transform = transforms.ToTensor()
            tensor_image = transform(image)

        # get the caption
        tokens = self.processor.clean_text(caption, self.cfg)
        tensor_cap_ids = [self.vocab.get(token, self.vocab[self.cfg.UNK_TOKEN]) for token in tokens]
        tensor_cap_ids = torch.LongTensor(tensor_cap_ids)
        # return also all captions in tensor
        """
        # get all the captions per the image
        captions = self.img_to_captions[fname]
        all_caps = []

        for cap in captions:
            tokens = self.processor.clean_text(cap, self.cfg)
            cap_ids = [self.vocab.get(token, self.vocab[self.cfg.UNK_TOKEN]) for token in tokens]
            all_caps.append(cap_ids)
        """
        cap_length = len(tensor_cap_ids)
        return tensor_image, tensor_cap_ids, cap_length
            
    def __len__(self):
        return len(self.data_list)

    
    
def collate_fn(data):
    ''' Pad the different sized caption ids with zeros till the same size, and put them in a batch ready for loading.
    @param: data (list of tuples) contains the 
    @return (images batch, caption ids batch, length of caption ids)
    '''
    # sorts captions in a descending order
    # then pads them with 0s so that they have a uniform length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images_list, captions_list, cap_lengths  = list(zip(*data))
    # stack the images along the channels dimensions [num_image*D, H, W]
    image_batch = torch.stack(images_list, dim=0)
    # pad the captions with zeros unill max length
    max_length = len(captions_list[0])
    lengths = [len(elem) for elem in captions_list]
    caption_batch = torch.zeros(len(data), max_length).long()
    for idx, (caption, length) in enumerate(zip(captions_list, lengths)):
        caption_batch[idx, :length] = caption
    # returns batch of images,  batch of padded captions, batch of original lengths of captions
    return image_batch , caption_batch , torch.LongTensor(cap_lengths) # shape [bs, 1]
            


def transform_albumentation():
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2
    
    transform = {'train' : albu.Compose([
            albu.Resize((224, 224)),
            #transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            #transforms.RandomHorizontalFlip(),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
            ToTensorV2()]),

        'test' : albu.Compose([
            albu.Resize((224, 224)),
            albu.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
            ToTensorV2()]),
        }



def get_loaders(cfg, cuda=True):

    print('Creating Data Loaders ...')
    # get data & data loaders
    if cfg.transform_input:
        # resnet expects H and W to be at least 224
        transform = {'train' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                                        ]),
                'test' : transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225))
                                    ]) 
                }
    else:
        transform = {'train' : transforms.Resize((256, 256)), #512, 512 did not work
                'test' : transforms.Resize((256, 256))} # 224, 224

    # Data pre-processing
    processor = Preprocessor(cfg)

    # setup data
    train_dataset = Flicker8K('train', cfg, processor, transform=transform['train'])
    val_dataset = Flicker8K('dev', cfg, processor, transform=transform['test'])
    test_dataset = Flicker8K('test', cfg, processor,  transform=transform['test'])

    
    # setup kwargs
    train_kwargs = {'batch_size': cfg.batch_size, 'collate_fn' : collate_fn, 'shuffle': True, 
                                            'drop_last': True}
    test_kwargs = {'batch_size': cfg.test_batch_size, 'collate_fn' : collate_fn,
                                        'drop_last' : True, 'shuffle': False}
    if cuda:
        cuda_kwargs = {'num_workers': cfg.num_workers,
                'pin_memory': True,
                }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # data loaders
    
    train_loader = DataLoader(dataset= train_dataset, **train_kwargs)
    val_loader = DataLoader(dataset=val_dataset, **test_kwargs)
    test_loader = DataLoader(dataset=test_dataset, **test_kwargs)

    return train_loader, val_loader, test_loader, processor



class ReverseTransform:
    def __init__(self, transform):
        pass


if __name__ == '__main__':

    cfg = yaml_load_with_wandb("config-defaults.yaml", use_wandb=False)

    processor = Preprocessor(cfg)
    device = torch.device("cpu")
    # Setup models
    encoder = Encoder(cfg, device).to(device)
    decoder = Decoder(cfg, processor.vocab, device).to(device)
    # Run inference
    encoder, decoder, enc_optimizer, dec_optimizer, epoch_num = load_checkpoint(cfg, encoder, decoder)
    
