"""
This file includes code adapted from https://github.com/nrimsky/CAA/blob/main/utils/tokenize.py
"""
from typing import List, Tuple, Optional


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def tokenize(
    tokenizer,
    system_prompt: str,
    model_name: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    model_name: the name of the model
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """
    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if is_first_message:
            if model_name == 'llama-2':
                dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
            elif model_name == 'mistral':
                dialog_content = system_prompt + '\n' + instruction.strip()
            else:
                raise SystemExit("Unsupported model name: ", model_name)
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )

            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens
