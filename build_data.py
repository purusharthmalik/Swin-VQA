import PIL
import copy
import transformers
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from randaugment import RandomAugment

class VQA_RAD_Dataset(Dataset):

    def __init__(self, data_dir, img_tokens=32, seq_len=512, vocab_size=32000, mode='Train', start=0, text_type='blank'):
        self.img_root = './Data/VQA_RAD_Images/'
        self.data = pd.read_csv(data_dir).iloc[start:]
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained('./Llama-2/', legacy=False)
        self.mode = mode
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type

        # Normalizing the images with their mean and standard deviation
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        # During inference, we do only want to resize and normalize the input
        if mode == 'Test':
            self.transform = transforms.Compose([
                transforms.Resize((self.H, self.W), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize
            ])

            if self.text_type == 'random':
                self.text_type = 'choice'
        
        else:
            # List of transforms 
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((self.H, self.W), scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                normalize
            ])
        
        self.mode = mode
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    # Returns the number of samples in the dataset
    def __len__(self):
        return len(self.data)
    
    # Creating multiple choice questions
    def random_answer(self, qu, ans):
        ans = str(ans)
        pre_text = 'Question: '+qu+'The Answer is:'
        final_o = 'Question:'+qu+'The Answer is:'+ans
        return pre_text, final_o
    
    # Loads and returns a sample from the dataset at the given index 'idx'
    def __getitem__(self, idx):
        # Getting the sample
        sample = self.data.iloc[idx]
        qu = sample['QUESTION']
        ans = sample['ANSWER']

        # Read image patches
        img_path = self.img_root + sample['IMAGEID'].split('/')[-1]
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        if self.mode == 'Train':
            pre_text, final_o = self.random_answer(qu, ans)
            # Creating tokens for QA
            final_o = self.tokenizer(final_o)
            # Storing only the input ids of the tokens
            input_ids = final_o['input_ids']
            # Adding the EOS token
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)

            # Padding the inputs if the length is less than the seq_len
            if len(input_ids) < self.seq_len:
                input_ids = np.pad(input_ids, (0, self.seq_len - len(input_ids)), 'constant', constant_values=0)
            # Otherwise, truncating the input
            else:
                input_ids = input_ids[:self.seq_len]

            # Copying the token ids
            label = copy.deepcopy(input_ids)
            # Changing the padding tokens to -100
            label[label==0] = -100
            if pre_text != '':
                pre_text = self.tokenizer(pre_text)
                # Checking if the answer if present or not
                if len(pre_text['input_ids'])<len(label):
                    # Highlighting the answer
                    label[:len(pre_text['input_ids'])] = -100
            label = label.tolist()
            # Concatenate the labels
            label = np.array(self.img_padding + label)

            item = {
                'input_ids': input_ids,
                'images': image,
                'labels': label
            }
        
        elif self.mode == 'Test':
            item = {
                'input_ids': 'Question: '+qu+'The Answer is: ',
                'img_path': sample['img_name'],
                'images': image,
                'labels': ans
            }

        # Return the data item
        return item