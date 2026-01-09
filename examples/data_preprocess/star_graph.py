import random
import json

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def convert_sample_to_prompt(sample: str):
    # Split input
    graph_part, rest = sample.split("/")
    query_part, path_part = rest.split("=")

    # Parse edges
    edges = []
    for e in graph_part.split("|"):
        u, v = e.split(",")
        edges.append((int(u), int(v)))

    # Parse start and target
    start, target = map(int, query_part.split(","))

    # Parse correct path
    path = path_part.strip()
    boxed_path = f"\\boxed{{{path}}}"

    # Question templates
    question_templates = [
        lambda e, s, t: (
            f"Consider a graph with the following undirected edges: "
            f"{', '.join(f'({u},{v})' for u, v in e)}. "
            f"Starting from node {s}, find a valid path to reach node {t}."
        ),
        lambda e, s, t: (
            f"A network is represented by nodes connected through undirected links. "
            f"The connections are {', '.join(f'{u}-{v}' for u, v in e)}. "
            f"If a signal starts at node {s}, how can it reach node {t}?"
        ),
        lambda e, s, t: (
            f"Let G be an undirected graph with edge set "
            f"{{{', '.join(f'({u},{v})' for u, v in e)}}}. "
            f"Find a path from vertex {s} to vertex {t}."
        ),
        lambda e, s, t: (
            f"You are given the following direct connections between points: "
            f"{', '.join(f'{u}-{v}' for u, v in e)}. "
            f"Starting at point {s}, find a sequence of connected points that leads to point {t}."
        )
    ]

    # Answer templates
    answer_templates = [
        lambda p: f"One valid route is {p}.",
        lambda p: f"A possible path is {p}.",
        lambda p: f"A correct sequence is {p}.",
        lambda p: f"One valid path from the start to the target is {p}."
    ]

    question = random.choice(question_templates)(edges, start, target)
    answer = random.choice(answer_templates)(boxed_path)

    return {
        "question": question,
        "answer": answer,
        "path": path
    }


# Example usage
if __name__ == "__main__":
    train_size = 12800 * 2
    train_data = [ ]
    with open("/home/nlp/hnn5071/next_token_data/datasets/graphs/deg_2_path_5_nodes_50_train_200000.txt", "r") as f:
        while len(train_data) < train_size:
            output = convert_sample_to_prompt(f.readline())
            train_data.append(output)

    test_size = 10000
    test_data = []
    with open("/home/nlp/hnn5071/next_token_data/datasets/graphs/deg_2_path_5_nodes_50_test_20000.txt", "r") as f:
        while len(test_data) < test_size:
            output = convert_sample_to_prompt(f.readline())
            test_data.append(output)

    # train_size = 7000
    # train_data = [ ]
    # with open("/home/nlp/hnn5071/next_token_data/datasets/graphs/deg_2_path_5_nodes_50_train_200000.txt", "r") as f:
    #     while len(train_data) < train_size:
    #         output = convert_sample_to_prompt(f.readline())
    #         train_data.append(output)

    # sample = train_data[0]
    # print(json.dumps(output, indent=2))

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # instruction_following = " Output the correct path within \\boxed{}."
    output_format_instruction = (
        " Express your final answer in the form "
        "\\boxed{v1,v2,...,vk}, where each vertex is separated by a comma."
    )
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer = example.pop("answer")
            path = example.pop("path")
            data_source = "star_graph"

            question = question_raw + output_format_instruction
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": path},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    print(len(train_dataset))
    # Sub-sample 30000 samples from train_dataset

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
