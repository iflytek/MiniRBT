import argparse
import json
from typing import List

from ltp import LTP
import os
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer

device = torch.device('cuda')


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def get_chinese_word(tokens: List[str]):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def add_sub_symbol(bert_tokens: List[str], chinese_word_set: set()):
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start: start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word


def prepare_ref(lines: List[str], ltp_tokenizer: LTP, bert_tokenizer: BertTokenizer):
    ltp_res = []
    for i in range(0, len(lines), 50):
        res = ltp_tokenizer.pipeline(lines[i: i + 50], tasks=["cws"])
        res = [get_chinese_word(r) for r in res.cws]
        ltp_res.extend(res)
    assert len(ltp_res) == len(lines)

    bert_res = []
    for i in range(0, len(lines), 50):
        res = bert_tokenizer(lines[i: i + 50], add_special_tokens=True)
        bert_res.extend(res["input_ids"])
    assert len(bert_res) == len(lines)

    ref_ids = []
    for input_ids, chinese_word in zip(bert_res, ltp_res):
        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        input_tokens = add_sub_symbol(input_tokens, chinese_word)
        ref_id = []
        # We only save pos of chinese subwords start with ##, which mean is part of a whole word.
        for i, token in enumerate(input_tokens):
            if token[:2] == "##":
                clean_token = token[2:]
                # save chinese tokens' pos
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    ref_id.append(i)
        ref_ids.append(ref_id)

    assert len(ref_ids) == len(bert_res)

    return ref_ids


def main(args):
    # For Chinese (Ro)Bert, the best result is from : RoBERTa-wwm-ext (https://github.com/ymcui/Chinese-BERT-wwm)
    # If we want to fine-tune these model, we have to use same tokenizer : LTP (https://github.com/HIT-SCIR/ltp)
    ltp_model_name_or_path = "./pretrained_model_path/ltp"
    # faster in GPU device
    ltp_tokenizer = LTP(pretrained_model_name_or_path=ltp_model_name_or_path)
    ltp_tokenizer.to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert)

    datadir = './dataset/train/train_files'
    for parent, dirnames, filenames in os.walk(datadir):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.readlines()
            save_path = os.path.join(
                "./dataset/train/ref_files", filename.split(".")[0]+"_ref.txt")
            data = [line.strip() for line in data if len(
                line) > 0 and not line.isspace()]  # avoid delimiter like '\u2029'
            ref_ids = prepare_ref(data, ltp_tokenizer, bert_tokenizer)
            save_file = open(save_path, "w", encoding="utf-8")
            data = [json.dumps(ref) + "\n" for ref in ref_ids]
            save_file.writelines(data)
            save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare_chinese_ref")
    parser.add_argument("--bert", type=str, default="./pretrained_model_path/RoBERTa",
                        help="resources for Bert tokenizer")

    args = parser.parse_args()
    main(args)
