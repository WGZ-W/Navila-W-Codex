#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import inspect
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from llava.mm_utils import process_images
from llava.model.action_head import OpenFlyActionHead, format_openfly_action_prompt
from llava.model.loss import soft_cross_entropy
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)
        self.action_head = None
        if getattr(config, "enable_action_head", False):
            self.initialize_action_head()

    def initialize_action_head(self) -> OpenFlyActionHead:
        """Create the OpenFly classifier and load it when resuming a checkpoint."""
        if self.action_head is not None:
            return self.action_head

        hidden_size = getattr(self.get_llm().config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("The language model config does not define hidden_size")
        self.config.enable_action_head = True
        self.action_head = OpenFlyActionHead(
            hidden_size=hidden_size,
            num_actions=getattr(self.config, "num_actions", 10),
            dropout=getattr(self.config, "action_head_dropout", 0.0),
        )

        reference_parameter = next(self.get_llm().parameters())
        if not reference_parameter.is_meta:
            self.action_head.to(device=reference_parameter.device, dtype=reference_parameter.dtype)

        resume_path = getattr(self.config, "resume_path", None)
        if resume_path:
            action_head_path = os.path.join(resume_path, "action_head", "pytorch_model.bin")
            if os.path.isfile(action_head_path):
                state_dict = torch.load(action_head_path, map_location="cpu")
                self.action_head.load_state_dict(state_dict)
        return self.action_head

    def get_action_head(self) -> Optional[OpenFlyActionHead]:
        return self.action_head

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        history_images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        action_labels: Optional[torch.LongTensor] = None,
        return_action_logits: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast, SequenceClassifierOutputWithPast]:
        self.freezed_module_patch()

        action_forward = action_labels is not None or return_action_logits
        if action_forward and self.get_action_head() is None:
            raise RuntimeError(
                "The OpenFly action head is disabled. Set enable_action_head=True in the model config before "
                "constructing the model."
            )

        # The expanded multimodal attention mask is required for selecting the
        # final real token after image embeddings have been inserted.
        if action_forward and attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                history_images,
            )

        if action_forward and attention_mask is None:
            sequence_shape = inputs_embeds.shape[:2] if inputs_embeds is not None else input_ids.shape
            attention_mask = torch.ones(sequence_shape, dtype=torch.bool, device=self.device)

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        if self.training and support_packing and not dpo_forward and not action_forward:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids

        if action_forward:
            base_outputs = self.get_llm().model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            action_logits = self.get_action_head()(base_outputs.last_hidden_state, new_attention_mask)
            action_loss = None
            if action_labels is not None:
                action_labels = action_labels.to(device=action_logits.device, dtype=torch.long).view(-1)
                if action_labels.shape[0] != action_logits.shape[0]:
                    raise ValueError(
                        f"Received {action_labels.shape[0]} action labels for a batch of {action_logits.shape[0]}"
                    )
                if ((action_labels < 0) | (action_labels >= self.config.num_actions)).any():
                    raise ValueError(f"Action labels must be in [0, {self.config.num_actions - 1}]")
                action_loss = F.cross_entropy(action_logits, action_labels)
                action_loss = action_loss * getattr(self.config, "action_loss_weight", 1.0)
            return SequenceClassifierOutputWithPast(
                loss=action_loss,
                logits=action_logits,
                past_key_values=base_outputs.past_key_values,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
            )

        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # print(outputs)

        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        # Loss rescale for SP & DP loss match
        if outputs.loss is not None and new_labels is not None:
            loss_weight = calculate_loss_weight(new_labels)
            outputs.loss = outputs.loss * loss_weight
        # outputs.loss = torch.nan_to_num(outputs.loss)
        # print(outputs.loss)

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs

    @torch.inference_mode()
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        history_images: Optional[Union[Image.Image, List[Image.Image], torch.FloatTensor]] = None,
        instruction: Optional[str] = None,
        return_action_id: bool = False,
        **kwargs,
    ) -> Union[int, np.ndarray]:
        """Predict one valid OpenFly action.

        Callers can provide already-tokenized NaVILA inputs, or pass one/more
        PIL images together with a navigation instruction.  The default return
        value is the eight-dimensional vector expected by OpenFly's evaluator.
        """
        if images is None and "pixel_values" in kwargs:
            images = kwargs.pop("pixel_values")
        # Accepted by OpenVLA's API but unnecessary for discrete classification.
        kwargs.pop("unnorm_key", None)
        kwargs.pop("do_sample", None)

        if input_ids is None:
            if image is None or instruction is None:
                raise ValueError("Provide either tokenized input_ids or both image and instruction")
            prompt_images = list(image) if isinstance(image, (list, tuple)) else [image]
            if history_images is None and len(prompt_images) > 1:
                history_images, prompt_images = prompt_images[:-1], prompt_images[-1:]
            conversation = [
                {
                    "from": "human",
                    "value": prompt_images + [format_openfly_action_prompt(instruction)],
                }
            ]
            media = extract_media(conversation, self.config)
            input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).unsqueeze(0)
            images = process_images(media["image"], self.get_vision_tower().image_processor, self.config)
            if history_images is not None and not torch.is_tensor(history_images):
                history_images = list(history_images) if isinstance(history_images, (list, tuple)) else [history_images]
                num_history_frames = int(getattr(self.config, "history_num_frames", 4))
                history_images = history_images[-num_history_frames:]
                if not history_images:
                    history_images = prompt_images
                history_images = [history_images[0]] * (num_history_frames - len(history_images)) + history_images
                history_images = self.get_vision_tower().image_processor.preprocess(
                    history_images, return_tensors="pt"
                )["pixel_values"].unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(self.device)
        if images is not None:
            images = images.to(device=self.device, dtype=self.dtype)
        if history_images is not None:
            history_images = history_images.to(device=self.device)

        outputs = self.forward(
            input_ids=input_ids,
            images=images,
            history_images=history_images,
            attention_mask=attention_mask,
            return_action_logits=True,
            use_cache=False,
        )
        action_id = int(outputs.logits.argmax(dim=-1)[0].item())
        if return_action_id:
            return action_id
        action_vector = self.get_action_head().action_ids_to_vectors(torch.tensor(action_id))
        return action_vector.cpu().numpy().astype(np.float32)


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
