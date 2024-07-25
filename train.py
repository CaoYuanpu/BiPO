# 0. imports
import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import BiPOTrainer, DPOConfig
from fastchat.conversation import get_conv_template

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_parameters(model):
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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, vec=None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        if vec is not None:
            self.vec = torch.nn.Parameter(vec)
        else:
            # Zero Init
            self.vec = torch.nn.Parameter(torch.zeros(4096))

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        output = (output[0]  +  (self.multiplier * self.vec),) + output[1:]
        return output

    def set_multiplier(self, multiplier):
        self.multiplier  = multiplier


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "we only support meta-llama/Llama-2-7b-chat-hf and mistralai/Mistral-7B-Instruct-v0.2"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    max_prompt_length: Optional[int] = field(default=2048, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=40, metadata={"help": "the number of training epochs"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[int] = field(default=15, metadata={"help": "the layer the steering vector extracted from"})

    # instrumentation
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def get_data(num_proc=1, behavior='power-seeking', train=True, template_name='llama-2'):
    if train:
        dataset = load_dataset("csv", data_files=f"./data/{behavior}/train.csv", split='train')
    else:
        dataset = load_dataset("csv", data_files=f"./data/{behavior}/test.csv", split='train')
    original_columns = dataset.column_names
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        prompt = []
        for question in samples["question"]:
            conv = get_conv_template(template_name)
            conv.set_system_message(SYSTEM_PROMPT)
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt.append(conv.get_prompt())
        return {
            "prompt": prompt,
            "chosen": [' ' + s for s in samples["matching"]],
            "rejected": [' ' + s for s in samples["not_matching"]],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(seed=11)
    if script_args.model_name_or_path not in ['meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.2']:
        print(f'{script_args.model_name_or_path} is not in supported model list. We support meta-llama/Llama-2-7b-chat-hf and mistralai/Mistral-7B-Instruct-v0.2')
    if script_args.model_name_or_path == 'meta-llama/Llama-2-7b-chat-hf':
        template_name = 'llama-2'
    elif script_args.model_name_or_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        template_name = 'mistral'
    print('[Behavior:] ', script_args.behavior, '[Layer:] ', script_args.layer, '[Model:] ', script_args.model_name_or_path)

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
    )
    model.model.layers[script_args.layer] = BlockWrapper(model.model.layers[script_args.layer])
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
    )
    print('-----------------------------')
    print(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    for name, param in model_ref.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if f'model.layers.{script_args.layer}.vec' not in name:
            param.requires_grad = False

    print('Finish loading pre-trained models...')

    # 2. Load training dataset
    train_dataset = get_data(behavior=script_args.behavior, train=True, template_name=template_name) 

    # 3. Load val dataset
    test_dataset = get_data(behavior=script_args.behavior, train=False, template_name=template_name) 
    
    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="epoch",
        output_dir="placeholder",
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

   # 5. initialize the DPO trainer
    dpo_trainer = BiPOTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset={'test_dataset_add': test_dataset, 'test_dataset_sub': test_dataset},
        tokenizer=tokenizer,
        behavior=script_args.behavior,
        layer=script_args.layer,
        name=template_name,
    )

    # 6. Start training
    print_trainable_parameters(model)
    dpo_trainer.train()
