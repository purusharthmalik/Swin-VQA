import torch
from model import VQAModel
from build_data import VQA_RAD_Dataset
import tqdm.auto as tqdm
from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader  

# Data classes for organizing model arguments
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    # Transformer attributes
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=12)
    img_token_num: Optional[int] = field(default=32)
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)
    checkpointing: Optional[bool] = field(default=True)
        
    # Image Encoder
    Vision_module: Optional[str] = field(default='CLIP')
    visual_model_path: Optional[str] = field(default='openai/clip-vit-base-patch32')
        
    # PEFT
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)

# Data classes for organizing data arguments
@dataclass
class DataArguments:
    pred_type: str = field(default='choice')
    train_csv_path: str = field(default='/kaggle/input/csv-data/test_data.csv', metadata={"help": "Path to the training data."})
    eval_csv_path: str = field(default='/kaggle/input/csv-data/test_data.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='meta-llama/Llama-2-7b-chat-hf', metadata={"help": "Path to the training data."})

# Data classes for organizing training arguments
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
def main():
    # Parsing command-line arguments using HF's argument parser
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    print("Setup Data")
    # Creating dataset objects for training and evaluation
    train_dataset = VQA_RAD_Dataset(data_args.train_csv_path, text_type = 'caption')
    eval_dataset = VQA_RAD_Dataset(data_args.eval_csv_path, text_type = 'caption')

    print("Setup Model")
    # Creating a model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel(model_args).to(device)
    
    print(model)

    # Setup for pre-training
    run_name_root = training_args.run_name
    output_dir_root = training_args.output_dir
    
    training_args.run_name = run_name_root+'_caption_pretrain'
    training_args.output_dir = output_dir_root + '/caption_pretrain/'
    
    print('Start Pretraining')
    # Creating a Trainer object
    trainer = Trainer(model=model, 
                      train_dataset = train_dataset, 
                      eval_dataset = eval_dataset,
                      args=training_args,
                      )
    # Attempting to resume from a checkpoint, otherwise start regular training
    try:
        trainer.train(resume_from_checkpoint=True)
    except:
        trainer.train()
    # Save the trainer's state
    trainer.save_state()
    
    print('Start training')  
    # Setup for main training
    training_args.run_name = run_name_root+'_' + data_args.pred_type + '_training'
    training_args.output_dir = output_dir_root + '/'+data_args.pred_type +'_training/'
    
    train_dataset.text_type = data_args.pred_type
    eval_dataset.text_type = data_args.pred_type
    
    # Creating dataloader objects
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    
    # Training loop with tqdm progress bar
    print("Training...")
    for epoch in tqdm.auto.trange(len(trainer), desc="Epoch"):
        trainer.train_step(train_dataset=train_dataloader)
    trainer.save_state()
    
if __name__ == "__main__":
    main()