# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" Utils for training
"""
import config
import logging
import os
import socket
import json
import numpy as np
import torch
from collections import abc

import logging
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def divide_parameters(named_parameters, lr=None):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_parameters_names = list(zip(
        *[(p, n) for n, p in named_parameters if not any((di in n) for di in no_decay)]))
    no_decay_parameters_names = list(
        zip(*[(p, n) for n, p in named_parameters if any((di in n) for di in no_decay)]))
    param_group = []
    if len(decay_parameters_names) > 0:
        decay_parameters, decay_names = decay_parameters_names
        # print ("decay:",decay_names)
        if lr is not None:
            decay_group = {'params': decay_parameters,
                           'weight_decay': config.args.weight_decay_rate, 'lr': lr}
        else:
            decay_group = {'params': decay_parameters,
                           'weight_decay': config.args.weight_decay_rate}
        param_group.append(decay_group)

    if len(no_decay_parameters_names) > 0:
        no_decay_parameters, no_decay_names = no_decay_parameters_names
        #print ("no decay:", no_decay_names)
        if lr is not None:
            no_decay_group = {'params': no_decay_parameters,
                              'weight_decay': 0.0, 'lr': lr}
        else:
            no_decay_group = {
                'params': no_decay_parameters, 'weight_decay': 0.0}
        param_group.append(no_decay_group)

    assert len(param_group) > 0
    return param_group


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


def MlmAdaptorWOLogitsMask(batch, model_outputs):
    return {'logits': (model_outputs["logits"]),
            'hidden': model_outputs["hidden_states"]}


def MlmAdaptorWithLogitsMask(batch, model_outputs):
    labels = batch["labels"]
    logits_mask = torch.ge(labels, -1)
    return {'logits': (model_outputs["logits"]),
            'hidden': model_outputs["hidden_states"],
            "logits_mask": logits_mask}


def RobertaForMlmAdaptorWithAtt(batch, model_outputs):
    labels = batch["labels"]
    logits_mask = torch.ge(labels, -1)
    # print(logits_mask)
    return {'logits': (model_outputs["logits"]),
            'hidden': model_outputs["hidden_states"],
            "attention": model_outputs["attentions"],
            "logits_mask": logits_mask}


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def move_to_device(batch, device):
    r"""Puts each data field to the device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return tuple(move_to_device(item, device) for item in batch)
    elif isinstance(batch, abc.Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    else:
        return batch
