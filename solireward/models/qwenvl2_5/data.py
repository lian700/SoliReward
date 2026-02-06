from typing import Any
from dataclasses import dataclass
from typing import List, Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
from qwen_vl_utils import process_vision_info
from collections import defaultdict
import torch
import time

@dataclass
class Qwen2_5VLDataCollator:
    processor: PreTrainedTokenizerBase

    def reorder_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        res = []
        for message in messages:
            new_message = defaultdict(list)
            role = message["role"]
            content = message["content"]

            new_message["role"] = role
            for c in content:
                type_c = c["type"]
                new_message["content"].append({
                    "type": type_c,
                    type_c: c[type_c]
                })

            res.append(new_message)
        return res

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Input: a list of dict, each dict is: 
        {
            "win": a OpenAI message list,
            "lose": a OpenAI message list,
            "meta": {
                "win": dict of meta info,
                "lose": dict of meta info,
            }
        }

        Return: a dict of:
        {
            "inputs_concat": BatchEncoding, # the concatenated inputs of win and lose
            "quality_concat": torch.Tensor, # the concatenated quality scores of win and lose, shape = (2 * batch_size,)
            "batch_size": int, # the batch size (number of pairs)
        }
        """

        messages_win = [item["win"] for item in features]
        messages_lose = [item["lose"] for item in features]
        messages_concat = messages_win + messages_lose
        messages_concat = [self.reorder_messages(messages) for messages in messages_concat]
        texts_concat = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_concat
        ]

        max_try = 10
        current_try = 0
        for _ in range(max_try):
            try:
                image_inputs_concat, video_inputs_concat = process_vision_info(messages_concat)
                break
            except RuntimeError as e:
                current_try += 1
                print(f"RuntimeError in process_vision_info, try {current_try}/{max_try}: {e}")
                time.sleep(1)
                if current_try >= max_try:
                    raise e

        inputs_concat = self.processor(
            text=texts_concat,
            images=image_inputs_concat,
            videos=video_inputs_concat,
            padding=True,
            return_tensors="pt",
        )

        quality_win = torch.tensor([item["meta"]["win"]["quality"] for item in features], dtype=torch.float32)
        quality_lose = torch.tensor([item["meta"]["lose"]["quality"] for item in features], dtype=torch.float32)
        quality_concat = torch.cat([quality_win, quality_lose], dim=0)

        res = {
            "inputs_concat": inputs_concat,
            "quality_concat": quality_concat,
            "batch_size": len(features),
            "win": {
                "quality": quality_win,
            },
            "lose": {
                "quality": quality_lose,
            },
        }

        return res
