import random
import json
import string

import argparse
import json
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

def generate_problem_description(sample: str):
    # Split graph and query
    graph_part, query_part = sample.split("/")

    # Parse relations
    relations = []
    for item in graph_part.split("|"):
        left, right = item.split("=")
        sign = right[0]
        parent = right[1:]
        relations.append((left, sign, parent))

    # Parse known value and query variable
    given, query = query_part.split(",")
    given_var, given_val = given.split("=")
    query_var = query.split("=")[0]

    given_val = int(given_val.replace("+", ""))

    # Sign relationship templates
    negative_templates = [
        "is the negative of ",
        # "is the opposite of ",
        "is the negation of ",
        "is equal to minus ",
        "is defined as the negative of ",
        "= -"
    ]

    positive_templates = [
        # "is the same as ",
        "is equal to ",
        "is identical to ",
        # "is defined as ",
        "is the positive of ",
        "= +"
    ]

    # Sentence components
    # relation_descriptions = [
    #     f"{u} is equal to {('+' if s == '+' else '-')}1 times {v}"
    #     for u, s, v in relations
    # ]   

    relation_descriptions = []
    for u, s, v in relations:
        if s == '-':
            s = random.choice(negative_templates)
        else:
            s = random.choice(positive_templates)
        # sentence = f"{u} is equal to {('+' if s == '+' else '-')}1 times {v}"
        sentence = f"{u} {s}{v}"
        relation_descriptions.append(sentence)

    # Problem templates
    templates = [
        lambda rels: (
            "Consider a system of variables where each variable is defined as either "
            "the positive or negative of another variable. The relationships are given as follows: "
            f"{'; '.join(rels)}. "
            f"If the value of {given_var} is {given_val}, determine the value of {query_var}."
        ),
        lambda rels: (
            "A dependency graph connects variables using edges with weights +1 or -1. "
            "Each edge indicates whether one variable equals another or its negation. "
            f"The relationships are: {', '.join(rels)}. "
            f"Given that {given_var} = {given_val}, find the value of {query_var}."
        ),
        lambda rels: (
            "Let each letter represent a numerical variable. Some variables are defined "
            "as the positive or negative of others according to the following rules: "
            f"{'; '.join(rels)}. "
            f"If {given_var} is assigned the value {given_val}, what is the resulting value of {query_var}?"
        ),
        lambda rels: (
            "You are given a signed graph where nodes represent variables and edges represent "
            "multiplicative relationships of +1 or -1. The rules are: "
            f"{', '.join(rels)}. "
            f"Assuming {given_var} = {given_val}, compute the value of {query_var}."
        )
    ]

    description = random.choice(templates)(relation_descriptions)
    return description


def generate_answer_description(solution: str):
    """
    Input format example:
    h=-c=-1,b=+h=-1,d=-b=+1,i=-d=-1
    """

    steps = solution.split(",")

    parsed_steps = []
    for step in steps:
        lhs, rhs, value = step.split("=")
        sign = rhs[0]
        parent = rhs[1:]
        value = int(value.replace("+", ""))
        parsed_steps.append((lhs, sign, parent, value))

    final_var, _, _, final_value = parsed_steps[-1]

    # Sign relationship templates
    negative_templates = [
        "is the negative of ",
        # "is the opposite of ",
        "is the negation of ",
        "is equal to minus ",
        "is defined as the negative of ",
        "= -"
    ]

    positive_templates = [
        # "is the same as ",
        "is equal to ",
        "is identical to ",
        # "is defined as ",
        "is the positive of ",
        "= +"
    ]

    step_sentences = []
    for lhs, sign, parent, value in parsed_steps:
        if sign == "-":
            relation = random.choice(negative_templates)
        else:
            relation = random.choice(positive_templates)
        sentence = f"{lhs} {relation}{parent}, so {lhs} = {value}"
        step_sentences.append(sentence)

    # Answer templates
    templates = [
        lambda steps_text: (
            f"Starting from the given value, we propagate the relationships step by step.\n"
            f"{'\n'.join(steps_text)}.\n"
            f"Therefore, the value of {final_var} is \\boxed{{{final_value}}}."
        ),
        lambda steps_text: (
            f"Using the dependency rules, we evaluate each variable in sequence.\n"
            f"{'\n'.join(steps_text)}.\n"
            f"Hence, {final_var} equals \\boxed{{{final_value}}}."
        ),
        lambda steps_text: (
            f"By following the signed relationships between variables, we obtain:\n"
            f"{'\n'.join(steps_text)}.\n"
            f"This shows that {final_var} has value \\boxed{{{final_value}}}."
        ),
        lambda steps_text: (
            f"Evaluating the variables according to their definitions gives:\n"
            f"{'\n'.join(steps_text)}.\n"
            f"As a result, the final answer is {final_var} = \\boxed{{{final_value}}}."
        )
    ]

    return random.choice(templates)(step_sentences)

# def convert_star_sample_to_legostar(sample: str):
#     # Split input
#     graph_part, rest = sample.split("/")
#     query_part, path_part = rest.split("=")

#     # Parse edges
#     edges = []
#     for e in graph_part.split("|"):
#         u, v = e.split(",")
#         edges.append((int(u), int(v)))



#     # Parse start and target
#     start, target = map(int, query_part.split(","))

#     # Parse correct path
#     path = path_part.strip()



#     return {
#         "question": question,
#         "path": path
#     }


def convert_graph_query_and_path(sample: str):
    """
    Input:
      32,3|16,12|3,19|32,34|34,6|6,16|19,47|47,28/32,12=32,34,6,16,12

    Output:
      weighted_graph / query solution_trace
    """

    graph_part, rest = sample.split("/")
    query_part, path_part = rest.split("=")

    # Parse edges
    edges = []
    nodes = set()
    for e in graph_part.split("|"):
        u, v = e.split(",")
        edges.append((u, v))
        nodes.add(u)
        nodes.add(v)

    # Parse query and path
    start, target = query_part.split(",")
    path_nodes = path_part.split(",")

    # Assign letters
    letters = list(string.ascii_lowercase)
    random.shuffle(letters)
    node_to_letter = {
        node: letters[i] for i, node in enumerate(sorted(nodes))
    }

    # Generate signed graph
    signed_edges = {}
    graph_output = []

    for u, v in edges:
        sign = random.choice(["+", "-"])
        lu, lv = node_to_letter[u], node_to_letter[v]
        signed_edges[(lu, lv)] = sign
        graph_output.append(f"{lv}={sign}{lu}")

    weighted_graph = "|".join(graph_output)

    # Convert query
    start_letter = node_to_letter[start]
    target_letter = node_to_letter[target]
    root_value = random.choice(['+', '-'])
    query_output = f"{start_letter}={root_value}1,{target_letter}=?"

    # Generate solution trace along the given path
    solution_steps = []
    current_value = 1 if root_value == '+' else -1 # start node value

    for i in range(len(path_nodes) - 1):
        u = node_to_letter[path_nodes[i]]
        v = node_to_letter[path_nodes[i + 1]]

        # Find sign (try both directions)
        if (u, v) in signed_edges:
            sign = signed_edges[(u, v)]
        else:
            sign = random.choice(["+", "-"])

        if sign == "+":
            next_value = current_value
            solution_steps.append(f"{v}=+{u}={'+' if next_value > 0 else '-'}1")
        else:
            next_value = -current_value
            solution_steps.append(f"{v}=-{u}={'+' if next_value > 0 else '-'}1")

        current_value = next_value

    solution_trace = ",".join(solution_steps)

    return f"{weighted_graph}/{query_output}", solution_trace

def convert_sample_to_prompt(sample: str):
    sample = sample.strip()
    query, solution_trace = convert_graph_query_and_path(sample)
    
    question = generate_problem_description(query)
    answer = generate_answer_description(solution_trace)

    return {
        "question": question,
        "answer": answer,
        
        "graph_query": query,
        "path": solution_trace
    }

# Example usage
if __name__ == "__main__":
    # sample = "32,3|16,12|3,19|32,34|34,6|6,16|19,47|47,28/32,12=32,34,6,16,12"
    # query, solution_trace = convert_graph_query_and_path(sample)

    # print(query, solution_trace)

    # # sample = "d=-b|f=-g|g=-e|a=+f|h=-c|i=-d|e=-c|b=+h/c=+1,i=?"
    # problem = generate_problem_description(query)
    # print(problem)

    # # solution = "h=-c=-1,b=+h=-1,d=-b=+1,i=-d=-1"
    # print(generate_answer_description(solution_trace))

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
        " Let's think step by step and output the final answer within \\boxed{}."
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
