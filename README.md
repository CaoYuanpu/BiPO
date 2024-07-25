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
