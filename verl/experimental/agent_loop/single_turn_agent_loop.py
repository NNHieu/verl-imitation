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
import copy
import logging
import os
from typing import Any
from uuid import uuid4

import math
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

import random

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))

        metrics = {}
        request_id = uuid4().hex

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
        )
        return output



@register("single_turn_agent_with_prefix")
class SingleTurnAgentWithPrefixLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        self.thinking_prefixes = []
        prefixes = self.config.data.get("thinking_prefix")
        
        self.probing_top_k_first_tokens = None
        if prefixes is not None and isinstance(prefixes, str):
            if "probing_and_uniform" in prefixes:
                self.probing_top_k_first_tokens = int(prefixes.split(":")[1])
            else:
                prefixes = prefixes.split("||")
                for prefix in prefixes:
                    if len(prefix) > 0:
                        self.thinking_prefixes.append(prefix)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))

        metrics = {}
        request_id = uuid4().hex

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
        
        prefix_ids = []
        is_validate = kwargs.get("_validate", False)
        if not is_validate and len(self.thinking_prefixes) > 0:
            chosen_prefix_str = random.choice(self.thinking_prefixes)
            if chosen_prefix_str != "no_prefix":
                prefix_ids = self.tokenizer.encode(chosen_prefix_str, add_special_tokens=False)
        
        with simple_timer("generate_sequences", metrics):
            if not is_validate and self.probing_top_k_first_tokens is not None:
                output = await self.server_manager.generate(
                    request_id=request_id, 
                    prompt_ids=prompt_ids, 
                    sampling_params=sampling_params, 
                    image_data=image_data, 
                    probing_topk_first_tokens = self.probing_top_k_first_tokens
                )
                candidate_ids = output.token_ids
                prefix_ids = [random.choice(candidate_ids)]
                print("Chosen prefix:", self.tokenizer.decode(prefix_ids))

            output = await self.server_manager.generate(
                request_id=request_id, 
                prompt_ids=prompt_ids, 
                sampling_params=sampling_params, 
                image_data=image_data, 
                prefix_ids = prefix_ids
            )
        
        # print("-----------------")
        # print(f"Outputs from single_turn_agent_with_prefix:")
        # print("Detokenized response:", self.tokenizer.decode(output.token_ids[:50]))
        # print("-----------------")
            
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
        )
        return output


@register("single_turn_competitive_agent")
class SingleTurnCompetitiveAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        self.perplexity_discard_threshold = self.config.actor_rollout_ref.rollout.agent.get("perplexity_discard_threshold", 16) # GPT2 perplexity
        
        self.thinking_prefixes = []
        prefixes = self.config.data.get("thinking_prefix")
        if prefixes is not None and isinstance(prefixes, str):
            prefixes = prefixes.split("||")
            for i, prefix in enumerate(prefixes):
                # assert len(prefix) > 0
                self.thinking_prefixes.append((i, prefix))
        assert len(self.thinking_prefixes) > 1


    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))

        metrics = {}
        request_id = uuid4().hex

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
        
        is_validate = kwargs.get("_validate", False)
        smallest_perplexity = 1

        with simple_timer("generate_sequences", metrics):
            prefix_ids = []
            if not is_validate:
                chosen_index, chosen_prefix_str = random.choice(self.thinking_prefixes)
                prefix_ids = self.tokenizer.encode(chosen_prefix_str, add_special_tokens=False)

            output = await self.server_manager.generate(
                request_id=request_id, 
                prompt_ids=prompt_ids, 
                sampling_params=sampling_params, 
                image_data=image_data, 
                prefix_ids = prefix_ids
            )

            if not is_validate:
                print(self.tokenizer.decode(output.token_ids[:20]))
                other_perplexities = []
                solution_ids = output.token_ids[len(prefix_ids):]
                for (other_index, other_prefix_str) in self.thinking_prefixes:
                    if other_index >= chosen_index:
                        continue
                    
                    print("Swaping from", chosen_prefix_str, "to", other_prefix_str)
                    other_prefix_ids = self.tokenizer.encode(other_prefix_str, add_special_tokens=False)
                    swap_prefix_output_token_ids = other_prefix_ids + solution_ids
                    swap_prefix_output_token_ids = swap_prefix_output_token_ids[:self.response_length]
                    
                    prob_request_id = uuid4().hex
                    logprob_output = await self.server_manager.generate(
                        request_id=prob_request_id, 
                        prompt_ids=prompt_ids + swap_prefix_output_token_ids, 
                        sampling_params=sampling_params, 
                        image_data=image_data,
                        compute_logprob_only=True,
                    )
                    assert len(logprob_output.log_probs) == (len(prompt_ids) + len(swap_prefix_output_token_ids))
                    swap_prefix_output_logprobs = logprob_output.log_probs[-len(solution_ids):]
                    swap_prefix_perplexity = math.exp(-sum(swap_prefix_output_logprobs) / len(solution_ids))
                    if math.isnan(swap_prefix_perplexity) or swap_prefix_perplexity < 1 or swap_prefix_perplexity > self.perplexity_discard_threshold:
                        print("Warning: swap_prefix_perplexity is NaN, < 1 or too hight")
                        swap_prefix_perplexity = 1
                    other_perplexities.append(swap_prefix_perplexity)
                
                print(other_perplexities)
                if len(other_perplexities) == 0:
                    smallest_perplexity = 1
                else:
                    smallest_perplexity = min(other_perplexities) - 1
        
        # print("-----------------")
        # print(f"Outputs from single_turn_agent_with_prefix:")
        # print("Detokenized response:", self.tokenizer.decode(output.token_ids[:50]))
        # print("-----------------")
            
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
            extra_fields = {
                "swap_prefix_perplexity": smallest_perplexity,
            }
        )
        return output
