import datasets
import os
from utils import *
import json
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
import argparse


def my_load_dataset(data_json, split, args, logger, tokenizer):
    chunk_map = ChunkTextMapBuilder("text", tokenizer)
    group_texts_map = GroupTextsRefsBuilder(args.max_seq_length)
    chunk_ref_map = ChunkRefMapBuilder("text")
    datasets_dict = []
    for idx, data_file in enumerate(data_json[split][split+"_files"]):
        filename = data_file.split("/")[-1].split(".")[0]
        cache_path = os.path.join(args.data_cache_dir, filename)
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(
                f'{split} datasets-{filename} has been loaded from disk')
        except Exception:
            logger.info(f"mapping {filename} of {split}_files")
            cache_dir = os.path.join(args.data_cache_dir, filename+"_text")
            os.makedirs(cache_dir, exist_ok=True)
            raw_dataset = load_dataset(
                "text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
            logger.info(f"{filename} of {split}_files has been loaded")
            # print(raw_dataset)
            tokenized_datasets = raw_dataset.map(
                chunk_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(
                    cache_dir, f'tokenized_{str(k)}.arrow') for k in raw_dataset}
            )
            # logger.info(tokenized_datasets.cache_files)
            # tokenized_datasets.save_to_disk(cache_path)
            # Add the chinese references if provided
            # print(tokenized_datasets)
            logger.info(f"{filename} of {split}_files has been tokenized")
            ref_cache_dir = os.path.join(
                args.data_cache_dir, filename+"_ref_text")
            os.makedirs(ref_cache_dir, exist_ok=True)
            ref_dataset = load_dataset(
                "text", data_files=data_json[split][split+"_ref_files"][idx], cache_dir=ref_cache_dir, keep_in_memory=False)
            logger.info(
                f"{filename}_ref_file of {split}_files has been loaded")
            # print(ref_dataset)  # DatasetDict
            ref_datasets = ref_dataset.map(
                chunk_ref_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(
                    ref_cache_dir, f'{filename}_ref_{str(k)}.arrow') for k in ref_dataset},
            )
            # logger.info(ref_datasets.cache_files)
            logger.info(
                f"{filename}_ref_file of {split}_files has been converted to an integer")
            # if len(tokenized_datasets["train"]["input_ids"]) != len(ref_datasets["train"]["refs"]):
            #     continue
            # print(ref_datasets)
            # print(tokenized_datasets["train"].features.type)
            # print(ref_datasets.features.type)
            tokenized_ref_cache_dir = os.path.join(
                args.data_cache_dir, filename+"tokenized_ref_text")
            os.makedirs(tokenized_ref_cache_dir, exist_ok=True)
            try:
                tokenized_ref_datasets = datasets.load_from_disk(
                    tokenized_ref_cache_dir, keep_in_memory=False)
            except Exception:
                logger.info(
                    f"{filename} tokenized datasets add chinese_refs column")
                tokenized_ref_datasets = tokenized_datasets["train"].add_column(
                    "chinese_ref", ref_datasets["train"]["refs"])
                tokenized_ref_datasets.save_to_disk(tokenized_ref_cache_dir)
                logger.info(
                    f"{filename} tokenized datasets add chinese_refs column done")
            tokenized_ref_datasets = datasets.load_from_disk(
                tokenized_ref_cache_dir, keep_in_memory=False)
            grouped_datasets = tokenized_ref_datasets.map(
                group_texts_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_name=os.path.join(
                    cache_dir, f'{filename}_{split}-grouped.arrow'),
                desc=f"Grouping {split}-{filename} texts in chunks of {args.max_seq_length}",)
            processed_dataset = grouped_datasets
            processed_dataset.save_to_disk(cache_path)
        if idx == 0:
            datasets_dict = processed_dataset
        else:
            assert datasets_dict.features.type == processed_dataset.features.type
            datasets_dict = concatenate_datasets(
                [datasets_dict, processed_dataset])

    return datasets_dict


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def is_existed_rec(name, obj):
    if isinstance(obj, list):
        for s in obj:
            assert(os.path.exists(s)), f"{name}:{s} does not exist"
            fsize = os.path.getsize(s)/float(1024*1024)
            print(s, round(fsize, 2))
    elif isinstance(obj, str):
        assert(os.path.exists(obj)), f"{name}:{obj} does not exist"
        fsize = os.path.getsize(obj)/float(1024*1024)
        print(obj, round(fsize, 2))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            is_existed_rec(name+'-'+k, v)
    else:
        print(f"Unknown object:{name}:{obj}")
        return


def read_data_json(filename: str):
    with open(filename, 'r') as f:
        data_json = json.load(f)
    # check all files exist
    for k, v in data_json.items():
        is_existed_rec(k, v)
    return data_json


## Map functions##
class ChunkTextMapBuilder:
    def __init__(self, text_column_name, tokenizer, max_seq_length=None):
        self.text_column_name = text_column_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        # Remove empty lines
        # examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return self.tokenizer(examples["text"], return_special_tokens_mask=True)


class ChunkRefMapBuilder:
    def __init__(self, text_column_name):
        self.text_column_name = text_column_name

    def __call__(self, examples):
        refs = {"refs": []}
        for item in examples["text"]:
            refs["refs"].append(json.loads(item))
        return refs


class GroupTextsRefsBuilder:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        # Concatenate all texts.
        # print(f"examples keys:{examples.keys()}")
        # print(len())
        firsts = {k: examples[k][0][0] for k in examples.keys()}
        lasts = {k: examples[k][0][-1] for k in examples.keys()}
        contents = {k: sum([vi[1:-1] for vi in v], [])
                    for k, v in examples.items() if k != "chinese_ref"}
        total_ref_length = 0
        contents["chinese_ref"] = []
        # contents["chinese_ref"] = [[]*len(examples["input_ids"])]
        for i, item in enumerate(examples["input_ids"]):
            # print(item)
            # print(examples["chinese_ref"][i])
            # contents["chinese_ref"][i] = [v + total_ref_length for v in examples["chinese_ref"][i]]
            contents["chinese_ref"].extend(
                [v + total_ref_length for v in examples["chinese_ref"][i]])
            total_ref_length += len(item) - 2
        # print(len(contents["chinese_ref"]))
        # exit(0)
        total_length = len(contents[list(examples.keys())[0]])

        content_length = self.max_seq_length - 2
        if total_length >= content_length:
            # add 1 here because we want to keep the short tails
            total_length = (total_length // content_length) * content_length
        # Split by chunks of max_len.
        result = {}
        ref_index = 0
        for k, t in contents.items():
            result[k] = []
            tmp = 0
            for i in range(0, total_length, content_length):
                if k == "chinese_ref":
                    ref_line = []
                    for ref in t[ref_index:]:
                        if ref <= i + content_length:
                            ref_line.append(ref-i)
                            tmp = ref
                        else:
                            break
                    ref_index = t.index(tmp)+1
                    result[k].extend([ref_line])
                    # print(len(result["chinese_ref"]))
                else:
                    result[k].extend(
                        [[firsts[k]] + t[i: i + content_length] + [lasts[k]]])
        return result


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger("my_dataset")
    tokenizer = BertTokenizer.from_pretrained(
        "./pretrained_model_path/RoBERTa")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cache_dir", default=None)
    parser.add_argument("--preprocessing_num_workers", default=10, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    args.preprocessing_num_workers = 20
    args.data_cache_dir = "./dataset"
    args.max_seq_length = 512
    data_json = read_data_json("./jsons/data.json")
    train_dataset = my_load_dataset(
        data_json, "train", args, logger, tokenizer)
    print("===========done==================")
