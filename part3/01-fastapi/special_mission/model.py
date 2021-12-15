import io
from typing import List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import tokenizers
from transformers.utils.dummy_pt_objects import AutoModel



def get_model(model_path: str = "ainize/klue-bert-base-mrc") -> AutoModelForQuestionAnswering.from_pretrained:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
    return model


def get_tokenizer(model_path : str = "ainize/klue-bert-base-mrc") -> AutoTokenizer.from_pretrained:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def _encode_texts(
    tokenizer: AutoTokenizer.from_pretrained,
    context: str,
    question: str
    ) -> Union[torch.tensor, torch.tensor]:
    encodings = tokenizer(context,
                          question,
                          max_length=512,
                          truncation=True,
                          padding="max_length",
                          return_token_type_ids=False)
    encodings = {key: torch.tensor([val]) for key, val in encodings.items()}
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    return input_ids, attention_mask


def predict_from_texts(
    context: str,
    question: str,
    model_path: str = "ainize/klue-bert-base-mrc"
    ) -> str:
    model = get_model(model_path)
    tokenizer = get_tokenizer(model_path)
    input_ids, attention_mask = _encode_texts(
        tokenizer=tokenizer,
        context=context,
        question=question
    )
    pred = model(input_ids, attention_mask=attention_mask)
    start_logits, end_logits = pred.start_logits, pred.end_logits
    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
    pred_ids = input_ids[0][token_start_index: token_end_index + 1]
    prediction = tokenizer.decode(pred_ids)
    
    return prediction


