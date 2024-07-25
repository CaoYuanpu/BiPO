"""
This file includes code adapted from https://github.com/nrimsky/CAA/blob/main/llama_wrapper.py
"""

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from tok import tokenize
from typing import Tuple, Optional
from fastchat.conversation import get_conv_template

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None

        self.save_internal_decodings = False
        self.do_projection = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (t.norm(
                last_token_activations
            )  * t.norm(self.calc_dot_product_with))
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            output = (output[0]  +  self.add_activations,) + output[1:]
        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False):
        self.add_activations = activations
        self.do_projection = do_projection

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []


class ModelWrapper:
    def __init__(
        self,
        token,
        system_prompt,
        model_name
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.model_name = model_name
        if self.model_name == 'llama-2':
            self.model_name_path = f"meta-llama/Llama-2-7b-chat-hf"
        elif self.model_name == 'mistral':
            self.model_name_path = f"mistralai/Mistral-7B-Instruct-v0.2"
        else:
            raise SystemExit("Unsupported model name: ", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, use_auth_token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, use_auth_token=token
        )
        self.model = self.model.to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        tokens = tokenize(
            self.tokenizer,
            self.system_prompt,
            self.model_name,
            [(prompt, None)],
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    
    def generate_text_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:       
        tokens = tokenize(
            self.tokenizer,
            self.system_prompt,
            self.model_name,
            history,
            no_final_eos=True,
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate_text_do_sample_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:       
        tokens = tokenize(
            self.tokenizer,
            self.system_prompt,
            self.model_name,
            history,
            no_final_eos=True,
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate_do_sample(tokens, max_new_tokens=max_new_tokens)
    
    def generate(self, tokens, max_new_tokens=50):
        with t.no_grad():
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0]

    def generate_do_sample(self, tokens, max_new_tokens=50):
        with t.no_grad():
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, temperature=2.0, top_p=1.0, do_sample=True
            )
            return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_logits_with_conversation_history(self, history: Tuple[str, Optional[str]]):
        tokens = tokenize(
            self.tokenizer,
            self.system_prompt,
            self.model_name,
            history,
            no_final_eos=True,
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, do_projection=False):
        self.model.model.layers[layer].add(activations, do_projection)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )


    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))

