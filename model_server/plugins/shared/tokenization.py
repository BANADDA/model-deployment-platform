# model_server/plugins/shared/tokenization.py

import torch
from typing import List, Dict

class TokenizerUtils:
    @staticmethod
    def truncate_to_max_length(tokens: List[int], max_length: int) -> List[int]:
        return tokens[:max_length]

    @staticmethod
    def pad_sequence(tokens: List[int], max_length: int, pad_token_id: int) -> List[int]:
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [pad_token_id] * (max_length - len(tokens))

    @staticmethod
    def create_attention_mask(tokens: List[int], pad_token_id: int) -> List[int]:
        return [1 if token != pad_token_id else 0 for token in tokens]