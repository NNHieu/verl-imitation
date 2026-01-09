# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

# from math_verify import parse

# def extract_solution(solution_str):
#     return remove_boxed(last_boxed_only_string(solution_str))

def extract_solution(solution_str, split):
    # solution = re.search("The answer is: (.+)$", solution_str)
    # assert solution is not None, solution_str
    # final_solution = solution.group(0)
    # final_solution = final_solution.split("The answer is: ")[1].replace(",", "")
    # parsed_final_solution = parse(final_solution)
    # assert len(parsed_final_solution) > 0, final_solution
    # parsed_final_solution = parsed_final_solution[0]
    # return parsed_final_solution
    if split == "train":
        search_pattern = "The answer is: "
    else:
        search_pattern = "#### "
    solution = re.search(f"{search_pattern}(\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split(search_pattern)[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "/data/hnn5071/lrm/datasets/mathinstruct-1/partialmutual_combine_metamath"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # if local_dataset_path is not None:
    #     dataset = datasets.load_dataset(
    #         local_dataset_path,
    #     )
    # else:
    #     dataset = datasets.load_dataset(
    #         data_source,
    #     )

    train_dataset = datasets.load_dataset("parquet", data_files=data_source + "/train.parquet")['train']
    test_dataset = datasets.load_dataset("openai/gsm8k", "main")["test"]

    instruction_following = " Let's think step by step and output the final answer within \\boxed{}."
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    print(train_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            if split == "test":
                question_raw = example.pop("question")
                answer_raw = example.pop("answer")
                data_source = "openai/gsm8k"
            else:
                example.pop("length")
                example.pop("gsm_idx")
                question_raw = example.pop("query")
                answer_raw = example.pop("response")
                data_source = "meta-math/MetaMathQA-40K"
            question = question_raw + instruction_following

            solution = extract_solution(answer_raw, split)
            if solution is None:
                # print("solution is None:", answer_raw)
                solution = "NA"

            if r"\boxed{" in answer_raw:
                answer_raw = "\n".join(answer_raw.split("\n")[:-1])
            else:
                answer_raw = "\n".join(answer_raw.split("\n")[:-1] + [f"The answer is \\boxed{{{solution}}}"])

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Filter out samples where ground truth solution is None
    train_dataset = train_dataset.filter(lambda example: example["reward_model"]["ground_truth"] is not None)
    test_dataset = test_dataset.filter(lambda example: example["reward_model"]["ground_truth"] is not None)

    # # Filter out samples where ground truth solution is None
    # train_dataset = train_dataset.filter(lambda example: "rephras" example["type"])
    # test_dataset = test_dataset.filter(lambda example: "rephras" example["type"])


    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    print(len(train_dataset), len(test_dataset))
    # Sub-sample 30000 samples from train_dataset
    if len(train_dataset) > 30000:
        train_dataset = train_dataset.shuffle(seed=42).select(range(30000))

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
