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
Preprocessing script before training the distilled model.
"""
import argparse

import torch

from transformers import BertForMaskedLM, RobertaForMaskedLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extraction some layers of the full RobertaForMaskedLM for Transfer Learned"," Distillation"
        ))
    parser.add_argument("--model_name_or_path", default="./pretrained_model_path/RoBERTa", type=str)
    parser.add_argument("--dump_checkpoint", default="./saves/init_checkpoint/init_checkpoint_S.pth", type=str)
    parser.add_argument("--vocab_transform", action="store_true")
    args = parser.parse_args()

    # model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
    # prefix = "roberta"
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    prefix = "bert"
    # print(dict(model.named_parameters()).keys())
    # print(dict(model.lm_head.named_parameters()).keys())

    state_dict = model.state_dict()
    compressed_sd = {}

    # Embeddings #
    for w in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
        param_name = f"{prefix}.embeddings.{w}.weight"
        compressed_sd[param_name] = state_dict[param_name]
    for w in ["weight", "bias"]:
        param_name = f"{prefix}.embeddings.LayerNorm.{w}"
        compressed_sd[param_name] = state_dict[param_name]

    # Transformer Blocks #
    std_idx = 0
    for teacher_idx in [0, 2, 4, 7, 9, 11]:
        for layer in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "attention.output.LayerNorm",
            "intermediate.dense",
            "output.dense",
            "output.LayerNorm",
        ]:
            for w in ["weight", "bias"]:
                compressed_sd[f"{prefix}.encoder.layer.{std_idx}.{layer}.{w}"] = state_dict[
                    f"{prefix}.encoder.layer.{teacher_idx}.{layer}.{w}"
                ]
        std_idx += 1
    # print(state_dict)
    
    # # Language Modeling Head ###s
    # for layer in ["lm_head.decoder.weight", "lm_head.bias"]:
    #     compressed_sd[f"{layer}"] = state_dict[f"{layer}"]
    # if args.vocab_transform:
    #     for w in ["weight", "bias"]:
    #         compressed_sd[f"lm_head.dense.{w}"] = state_dict[f"lm_head.dense.{w}"]
    #         compressed_sd[f"lm_head.layer_norm.{w}"] = state_dict[f"lm_head.layer_norm.{w}"]

    print(f"N layers selected for distillation: {std_idx}")
    print(f"Number of params transferred for distillation: {len(compressed_sd.keys())}")

    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(compressed_sd, args.dump_checkpoint)
