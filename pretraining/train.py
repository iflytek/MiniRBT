# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training the distilled model.
"""
import logging
import config
import os
import time
import numpy as np
import torch
import random
from textbrewer import DistillationConfig, TrainingConfig, GeneralDistiller
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    BertConfig,
    RobertaForMaskedLM,
    BertModel,
    BertForMaskedLM,
    BertTokenizer,
    RobertaTokenizer,
    DataCollatorForWholeWordMask,
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from my_datasets import my_load_dataset
from utils import *
from functools import partial
# from accelerate import Accelerator


def args_check(logger, args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.warning("Output directory () already exists and is not empty.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        logger.info("rank %d device %s n_gpu %d distributed training %r",
                    torch.distributed.get_rank(), device, n_gpu, bool(args.local_rank != -1))
    args.n_gpu = n_gpu
    args.device = device
    return device, n_gpu


def main():
    config.parse()
    args = config.args

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger = logging.getLogger("Main")
    # logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # arguments check
    device, n_gpu = args_check(logger, args)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank != -1:
        logger.warning(
            f"Process rank: {torch.distributed.get_rank()}, device : {args.device}, n_gpu : {args.n_gpu}, distributed training : {bool(args.local_rank!=-1)}")

    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")
    # set seeds
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    forward_batch_size = int(args.train_batch_size /
                             args.gradient_accumulation_steps)
    args.forward_batch_size = forward_batch_size

    # TOKENIZER #
    tokenizer = BertTokenizer.from_pretrained(args.teacher_name_or_path)
    # tokenizer = RobertaTokenizer.from_pretrained(args.teacher_name_or_path)
    stu_config = BertConfig.from_pretrained(args.student_config)
    stu_config.output_hidden_states = True
    # stu_config.output_attentions = args.output_attention_layers
    # model_S = RobertaForMaskedLM(stu_config)
    model_S = BertForMaskedLM(stu_config)
    logger.info(stu_config)
    logger.info("------total Number of parameters model_S: %i M" %
                (sum(p.numel() for p in model_S.parameters())//1000000))

    logger.info("model_S created.")
    if args.init_checkpoint_S is not None:
        logger.info(
            f"Loading pretrained weights from {args.init_checkpoint_S}")
        # model_S = RobertaForMaskedLM.from_pretrained(args.init_checkpoint_S, config=stu_config)
        state_dict_S = torch.load(args.init_checkpoint_S, map_location='cpu')
        missing_keys, unexpected_keys = model_S.load_state_dict(
            state_dict_S, strict=False)
        logger.info(f"missing keys:{missing_keys}")
        logger.info(f"unexpected keys:{unexpected_keys}")
    else:
        logger.info("Model_S is randomly initialized.")

    # TEACHER #
    # model_T = RobertaForMaskedLM.from_pretrained(args.teacher_name_or_path, output_hidden_states=True)
    model_T = BertForMaskedLM.from_pretrained(
        args.teacher_name_or_path, output_hidden_states=True)
    # model_T = BertForMaskedLM.from_pretrained(args.teacher_name_or_path, output_hidden_states=True, output_attentions = args.output_attention_layers)
    logger.info("model_T loaded.")
    model_T.to(device)
    model_S.to(device)

    params = list(model_S.named_parameters())
    all_trainable_params = divide_parameters(
        params, lr=args.learning_rate)
    logger.info("Length of all_trainable_params: %d",
                len(all_trainable_params))
    # DATA LOADER #
    data_json = read_data_json(args.data_files_json)
    train_dataset = my_load_dataset(
        data_json, "train", args, logger, tokenizer)

    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.forward_batch_size, drop_last=True, collate_fn=data_collator)

    logger.info(f"Loading student config from {args.student_config}")
    optimizer = AdamW(all_trainable_params,
                      lr=args.learning_rate, correct_bias=False)
    if args.official_schedule == 'const':
        scheduler_class = get_constant_schedule_with_warmup
        scheduler_args = {'num_warmup_steps': int(
            args.warmup_proportion*args.num_train_steps)}
    elif args.official_schedule == 'linear':
        scheduler_class = get_linear_schedule_with_warmup
        scheduler_args = {'num_warmup_steps': int(
            args.warmup_proportion*args.num_train_steps), 'num_training_steps': args.num_train_steps}
    else:
        raise NotImplementedError
    logger.warning("***** Running training *****")
    logger.warning("local_rank %d Num split examples = %d",
                   args.local_rank, len(train_dataset))
    logger.warning("local_rank %d Forward batch size = %d",
                   args.local_rank, forward_batch_size)
    logger.warning("local_rank %d Num backward steps = %d",
                   args.local_rank, args.num_train_steps)

    ########### DISTILLATION ###########
    train_config = TrainingConfig(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ckpt_steps=args.ckpt_steps,
        log_dir=args.output_dir,
        output_dir=args.output_dir,
        device=args.device,
        fp16=args.fp16,
        local_rank=args.local_rank)
    logger.info(f"{train_config}")

    from matches import matches
    intermediate_matches = None
    if isinstance(args.matches, (list, tuple)):
        intermediate_matches = []
        for match in args.matches:
            intermediate_matches += matches[match]
    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches)

    adaptor_T = MlmAdaptorWithLogitsMask
    adaptor_S = MlmAdaptorWithLogitsMask

    distiller = GeneralDistiller(train_config=train_config,
                                 distill_config=distill_config,
                                 model_T=model_T, model_S=model_S,
                                 adaptor_T=adaptor_T,
                                 adaptor_S=adaptor_S)
    with distiller:
        distiller.train(optimizer, scheduler_class=scheduler_class,
                        scheduler_args=scheduler_args,
                        max_grad_norm=1.0,
                        dataloader=train_dataloader,
                        num_steps=args.num_train_steps)


if __name__ == "__main__":
    main()
