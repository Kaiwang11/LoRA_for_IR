from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config,BitsAndBytesConfig
import json
from typing import List, Dict, Optional, Union, Tuple
import os
from peft import LoraConfig, get_peft_model,AdaLoraConfig
import torch
class LoraTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None,
                 lora_config:Dict={},dora: bool=False,vera:bool=False ):
        

        super(LoraTransformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        self.dora=dora
        self.vera=vera
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model(model_name_or_path, config,lora_config, cache_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        #No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


    def _load_model(self, model_name_or_path, config,lora_config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     # bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True
            # )
            model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
            # model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, quantization_config=quantization_config)
            print("Original model")
            self.print_trainable_parameters(model)

            self.auto_model=get_peft_model(model,lora_config)
            print("Lora model")
            self.print_trainable_parameters(self.auto_model)
    
    def print_trainable_parameters(self,model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
        if self.dora==True:
            modules=[name for name,param in self.auto_model.named_parameters()  if param.requires_grad and 'lora_magnitude_vector' in name and  str(self.num_hidden_layers-1) in name]

            mag_weight=self.auto_model.get_submodule(modules[-1][:-8]).default
        elif self.vera==True:
            modules=[name for name,param in self.auto_model.named_parameters()  if param.requires_grad and 'vera_lambda_b' in name and  str(self.num_hidden_layers-1) in name]

        else:
            modules=[name for name,param in self.auto_model.named_parameters() if param.requires_grad and   'lora_A' in name and str(self.num_hidden_layers-1) in name]

        A_qlin_weight=self.auto_model.get_submodule(modules[0][:-7]).weight
        A_klin_weight=self.auto_model.get_submodule(modules[1][:-7]).weight
        A_vlin_weight=self.auto_model.get_submodule(modules[2][:-7]).weight
        A_outlin_weight=self.auto_model.get_submodule(modules[3][:-7]).weight

        if A_qlin_weight.sum()==0:
            A_qlin_weight=torch.ones_like(A_qlin_weight)
        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]
        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]
        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        lora_config_path='/tmp2/ikwang/beir/examples/retrieval/training/config/LoRA.json'
        # with open(lora_config_path) as fIn:
        #     lora_config_dict = json.load(fIn)
        # lora_config = LoraConfig(lora_config_dict)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_lin","k_lin","v_lin","out_lin"],
            lora_dropout=0.1,
            bias="none", #‘none’, ‘all’ or ‘lora_only'
            # modules_to_save=["decode_head"],
        )
        return LoraTransformer(model_name_or_path=input_path,lora_config=lora_config)# **config)
    






