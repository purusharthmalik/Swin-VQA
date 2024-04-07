from decoder import *
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange

import transformers
from transformers import CLIPVisionConfig
import torchvision.models as models
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()

@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)
    mapping_hidden_dim: int = field(default=1024)

def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config

class VQAModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.hidden_dim = model_args.hidden_dim
        self.cov_size = model_args.voc_size
        self.img_tokens = model_args.img_token_num
        self.H = model_args.H
        self.N = model_args.N
        self.Vision_module = model_args.Vision_module

        # Handling the image
        if self.Vision_module == "CLIP":
            self.vision_model = transformers.CLIPVisionModel.from_pretrained(model_args.visual_model_path,ignore_mismatched_sizes=True)
            num_ftrs = 768

        # Handling the question
        self.query_embed = nn.Embedding(self.img_tokens, num_ftrs) 
        
        decoder_layer = TransformerDecoderLayer(num_ftrs, self.H, 1024,
                                        0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(num_ftrs)
        self.decoder = TransformerDecoder(decoder_layer, self.N , decoder_norm,
                                  return_intermediate=False)
        
        # Fully Connected Layers
        self.fc_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2 = nn.Linear(num_ftrs, self.hidden_dim)

        self.llamacausal = self.Setup_model(model_args)

    def Setup_model(self, model_args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Llama-2")
        model = transformers.LlamaForCausalLM.from_pretrained(model_args.model_path, token="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",
                                                              force_download=True, resume_download=False)
        model.to(device)
        print("Onto PEFT")

        if model_args.checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
        if model_args.is_lora:
            print("Setup PEFT")
            peft_config = get_peft_config(peft_args=model_args)
            model = get_peft_model(model, peft_config)
        return model
    
    def image_encoder(self, pxl):
        out_emb = self.vision_model(pixel_values=pxl)['last_hidden_state'][:,1:,:]
        return out_emb
    
    def forward(self,input_ids,images,labels):
        
        B = images.shape[0]
        # Image encoding
        x = self.image_encoder(images)
        features = x.transpose(0,1)
        
        # Encoding the question
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
        features,ws = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        features = features.transpose(0,1)
        
        # Fully connected layers 
        features = rearrange(features,'b n d  -> (b n) d')
        features = self.fc_l1(features)
        features = F.relu(features)
        features = self.fc_l2(features)
        features = rearrange(features,'(b n) d -> b n d',b=B)
        
        
        # LLM
        input_embedding = self.llamacausal.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([features,input_embedding], dim=1)
        
        output = self.llamacausal(inputs_embeds = input_embedding, labels = labels)
        
        # Calculating the loss
        loss = CrossEntropyLoss(output.view(-1, self.llamacausal.config.vocab_size), labels.view(-1))
        
        return loss
    
    def generate(self,input_ids,images):
        with torch.no_grad():
            B = images.shape[0]
            # Image encoding
            x = self.image_encoder(images)
            features = x.transpose(0,1)
            
            # Question encoding
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # query_number, batch, dim
            features,ws = self.decoder(query_embed, features, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
            features = features.transpose(0,1)
            features = rearrange(features,'b n d  -> (b n) d')
            features = self.fc_l1(features)
            features = F.relu(features)
            features = self.fc_l2(features)
            features = rearrange(features,'(b n) d -> b n d',b=B)
            ### LLM ###
            input_embedding = self.llamacausal.get_input_embeddings()(input_ids)
            input_embedding = torch.cat([features,input_embedding], dim=1)
            
            generation = self.llamacausal(inputs_embeds = input_embedding)['logits']
            return generation