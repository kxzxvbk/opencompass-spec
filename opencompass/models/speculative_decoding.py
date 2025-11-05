import re
from typing import List, Union, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from volcenginesdkarkruntime import Ark

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


large_model_sp = """
Your task is to verify the draft in <draft> tag. NEVER try to complete the draft text, only care about the correctness.
1. If correct, always return: <draft>No Error</draft>
2. Else, return: <draft>corrected draft</draft>. DO NOT return full text, rewrite the complete draft part.
""".strip()


def run_logitless_speculative_decoding(
    tokenizer, 
    small_model, 
    openai_client, 
    large_model_name, 
    message: list[dict],
    draft_length: int = 64,
    max_new_tokens: int = 512,
    use_spec: bool = True,
    temperature: float = 0.6,
):
    """
    Run speculative decoding.
    """
    
    full_text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    input_text = full_text
    generated_token_count = 0
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    while generated_token_count < max_new_tokens:
        input_ids = tokenizer.encode(full_text, return_tensors="pt").to("cuda")

        # Generate draft.
        with torch.no_grad():
            draft_output_ids = small_model.generate(
                input_ids,
                max_new_tokens=draft_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )

        draft_token_ids = draft_output_ids[0, input_ids.shape[1]:].tolist()
        draft_text = tokenizer.decode(draft_token_ids, skip_special_tokens=True)
        
        # Generation ends when draft is empty.
        if len(draft_text) == 0:
            break

        # If not use speculative decoding, append draft to full text.
        if not use_spec:
            full_text += draft_text
            generated_token_count += len(draft_token_ids)
            continue
        
        # Call large model to verify draft.
        response = openai_client.chat.completions.create(
            model=large_model_name,
            messages=[
                {"role": "system", "content": large_model_sp},
                {"role": "user", "content": full_text + f"<draft>{draft_text}</draft>"}
            ],
            temperature=0.3,
            max_completion_tokens=draft_length,
            thinking={"type": "disabled"}
        )
        large_model_completion_text = response.choices[0].message.content
        
        # Parse large model response.
        try:
            match = re.search(r"<draft>(.*?)</draft>", large_model_completion_text, re.DOTALL).group(1)
        except:
            match = "No Error"
        
        # If draft is correct, append to full text.
        if match == "No Error" or match.startswith(draft_text) or draft_text.startswith(match):
            full_text += draft_text
            generated_token_count += len(draft_token_ids)
        else:
            # Draft is not correct, append corrected draft to full text.
            corrected_draft = match
            full_text += corrected_draft
            generated_token_count += len(tokenizer.encode(corrected_draft, add_special_tokens=False))

    pure_input_ids = tokenizer.encode(input_text, return_tensors="pt")
    total_output_ids = tokenizer.encode(full_text, return_tensors="pt")
    generated_token_ids = total_output_ids[0, pure_input_ids.shape[1]:]
    return tokenizer.decode(generated_token_ids, skip_special_tokens=True)


class SpecModel(BaseModel):

    def __init__(
        self,
        path: str,
        api_key: str,
        large_model_name: str,
        draft_length: int = 64,
        use_spec: bool = True,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        meta_template: Optional[Dict] = None,
    ):  # noqa
        assert max_batch_size == 1, "SpecModel only supports max_batch_size=1."
        self._load_model(path=path)
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key,
            timeout=500,
        )
        
        self.draft_length = draft_length
        self.use_spec = use_spec
        self.max_seq_len = max_seq_len
        self.logger = get_logger()
        self.large_model_name = large_model_name
        self.template_parser = APITemplateParser(meta_template)

    def _load_model(
            self,
            path: str,
        ):
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    def generate(
            self,
            inputs: List[PromptType],
            max_out_len: int = 512,
            temperature: float = 0.6
        ) -> str:
        
        # Add default max_out_len.
        if max_out_len is None:
            max_out_len = 4096

        dialogs = []
        results = []
        for input in inputs:
            assert isinstance(input, (str, PromptList))
            if isinstance(input, str):
                dialog = [{'role': 'user', 'content': input}]
            else:
                dialog = []
                for item in input:
                    msg = {'content': item['prompt']}
                    if item['role'].upper() == 'HUMAN':
                        msg['role'] = 'user'
                    elif item['role'].upper() == 'BOT':
                        msg['role'] = 'assistant'
                    elif item['role'].upper() == 'SYSTEM':
                        msg['role'] = 'system'
                    else:
                        raise ValueError(f'Unknown role: {item["role"]}')
                    dialog.append(msg)
            dialogs.append(dialog)
            res = run_logitless_speculative_decoding(
                self.tokenizer, 
                self.model, 
                self.client, 
                self.large_model_name, 
                dialog,
                draft_length=self.draft_length,
                max_new_tokens=max_out_len,
                use_spec=self.use_spec,
                temperature=temperature,
            )
            results.append(res)
        return results

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, bos=True, eos=True)) + 100


if __name__ == "__main__":
    # Test the SpecModel.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--local_model", type=str, required=True)
    parser.add_argument("--large_model_name", type=str, default="deepseek-v3-1-250821")
    parser.add_argument("--disable_spec", action="store_true")
    args = parser.parse_args()
    model = SpecModel(
        path=args.local_model,
        api_key=args.api_key,
        large_model_name=args.large_model_name,
        use_spec=not args.disable_spec,
    )
    res = model.generate([args.prompt], max_out_len=2048)
    print(res)
