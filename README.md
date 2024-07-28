# Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization


 This project proposes an approach that could produce more effective steering vectors through **Bi**-directional **P**reference **O**ptimization (**BiPO**). It is designed to allow steering vectors to directly influence the generation probability of contrastive human preference data pairs, thereby offering a more precise representation of the target behavior. By carefully adjusting the direction and magnitude of the steering vector, one can enable personalized control over the desired behavior across a spectrum of intensities. BiPO can be applied to various open-ended generation tasks, including **steering AI personas**, **managing truthfulness**, **mitigating hallucinations**, and **addressing jailbreaking attacks**.

 *   Paper: [https://arxiv.org/pdf/2406.00045](https://arxiv.org/pdf/2406.00045)

 ## Cite this Work
If you find this project helpful in your research, please cite our paper:

Cao, Yuanpu et al. “Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization.” Arxiv (2024).

```bibtex
@article{cao2024personalized,
  title=    {Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization},
  author=   {Cao, Yuanpu and Zhang, Tianrong and Cao, Bochuan and Yin, Ziyi and Lin, Lu and Ma, Fenglong and Chen, Jinghui},
  journal=  {arXiv preprint arXiv:2406.00045},
  year=     {2024}
}
```

## Installation
First, clone the repo with
```
git clone https://github.com/CaoYuanpu/BiPO.git
```

Install dependencies:
```
cd BiPO
pip3 install -r requirements.txt
```

## Training steering vector
```
accelerate launch --gpu_ids <your gpu ids> train.py
```
Please note that all our experiments are conducted on a single GPU. Some important hyperparameters are as follows.
*   `--behavior`: the behavior that the user intends to steer
*   `--num_train_epochs`: the number of training epochs, default 100
*   `--learning_rate`: optimizer learning rate, default 5e-4
*   `--per_device_train_batch_size`: train batch size per device, default 4
*   `--model_name_or_path`: supported models, including meta-llama/Llama-2-7b-chat-hf and mistralai/Mistral-7B-Instruct-v0.2
*   `--layer`: the layer the steering vector extracted from

For example, the following two commands train vectors to steer a power-seeking persona on Llama-2 and Mistral, respectively:
```
accelerate launch --gpu_ids 0 train.py --layer 15 --behavior power-seeking --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```
```
accelerate launch --gpu_ids 0 train.py --layer 13 --behavior power-seeking --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2
```

Training vectors to steer the wealth-seeking persona:
```
accelerate launch --gpu_ids 0 train.py --layer 15 --behavior wealth-seeking --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```
```
accelerate launch --gpu_ids 0 train.py --layer 13 --behavior wealth-seeking --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2
```

Training vector to steer hallucination on Llama-2-7b-chat:
```
accelerate launch --gpu_ids 0 train.py --layer 15 --behavior hallucination --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```

Training vector to steer jailbreaking on Llama-2-7b-chat:
```
accelerate launch --gpu_ids 0 train.py --layer 15 --behavior jailbreak --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```

We also provide pretrained vectors in [**pretrained_vector/**](https://github.com/CaoYuanpu/BiPO/tree/main/pretrained_vector)

## Steering model generation on open-ended tasks
```
python prompting_with_steering.py
```
Some important hyperparameters are as follows.
*   `--behavior`: the behavior that the user intends to steer
*   `--model_name`: llama-2 / mistral
*   `--layer`: the layer the steering vector extracted from
*   `--ckp_epoch`: the epoch the vector checkpoint obtained from
*   `--multipliers`: a list of multipliers that be used to adjust the magnitude of the vectors
*   `--max_new_tokens`: the maximum number of tokens that the model is allowed to generate in the output.
*   `--pretrained`: whether to use a pretrained vector or not
*   `--verbose`: Whether to print the generated output or not

For example, we can steer power-seeking persona using pretrained vectors as follows:
```
python prompting_with_steering.py --behavior power-seeking --model_name llama-2 --layer 15 --ckp_epoch 20 --multipliers 2.0 -2.0 1.0 -1.0 0 --pretrained --verbose
```

```
python prompting_with_steering.py --behavior power-seeking --model_name mistral --layer 13 --ckp_epoch 5 --multipliers 2.0 -2.0 1.0 -1.0 0 --pretrained --verbose
```

Steering wealth-seeking persona:
```
python prompting_with_steering.py --behavior wealth-seeking --model_name llama-2 --layer 15 --ckp_epoch 20 --multipliers 2.0 -2.0 1.0 -1.0 0 --pretrained --verbose
```

```
python prompting_with_steering.py --behavior wealth-seeking --model_name mistral --layer 13 --ckp_epoch 5 --multipliers 2.0 -2.0 1.0 -1.0 0 --pretrained --verbose
```

Steering hallucination of Llama-2-7b-chat:
```
python prompting_with_steering.py --behavior hallucination --model_name llama-2 --layer 15 --ckp_epoch 20 --multipliers 2.0 -2.0 1.0 -1.0 0 --pretrained --verbose
```

Steering jailbreaking of Llama-2-7b-chat:
```
python prompting_with_steering.py --behavior jailbreak --model_name llama-2 --layer 15 --ckp_epoch 10 --multipliers 1.0 --pretrained --verbose
```

## Evaluation with GPT-4 judge
After obtaining the generation, we can evaluate the steering effects using LLMs-as-judge by running:

```
python evaluate.py
```
Some important hyperparameters are as follows.
*   `--judge`: judge model, including gpt-4o/gpt-4o-mini/gpt-4-0125-preview
*   `--api_key`: your openai api key
*   `--behavior`: steered behavior
*   `--model_name`: llama-2 / mistral
*   `--layer`: the layer the steering vector extracted from
*   `--ckp_epoch`: the epoch the vector checkpoint obtained from
*   `--multipliers`: a list of multipliers that be used to adjust the magnitude of the vectors during generation process
*   `--verbose`: Whether to print the evaluation result or not

The following is an example to evaluate the average hallucination score of the steered generation:
```
python evaluate.py --judge gpt-4o-mini --behavior hallucination --model_name llama-2 --layer 15 --multipliers 2 --ckp_epoch 20 --api_key <your openai api key> --verbose
```