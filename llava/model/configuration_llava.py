# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        fps=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=True,
        mm_projector_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        num_time_tokens=None,
        time_token_format=None,
        enable_action_head=False,
        num_actions=10,
        action_head_dropout=0.0,
        action_loss_weight=1.0,
        enable_history_mamba=False,
        history_num_frames=4,
        history_image_size=384,
        history_patch_size=32,
        history_hidden_size=768,
        history_depth=4,
        **kwargs,
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format
        self.enable_action_head = enable_action_head
        self.num_actions = num_actions
        self.action_head_dropout = action_head_dropout
        self.action_loss_weight = action_loss_weight
        self.enable_history_mamba = enable_history_mamba
        self.history_num_frames = history_num_frames
        self.history_image_size = history_image_size
        self.history_patch_size = history_patch_size
        self.history_hidden_size = history_hidden_size
        self.history_depth = history_depth
