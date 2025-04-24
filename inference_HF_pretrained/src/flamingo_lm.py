# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch.nn as nn

try:
    from .helpers import GatedCrossAttentionBlock
    from .utils import getattr_recursive, setattr_recursive
except:
    from helpers import GatedCrossAttentionBlock
    from utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer_sound, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.gated_cross_attn_layer_sound = gated_cross_attn_layer_sound
        self.decoder_layer = decoder_layer
        self.audio_x = None
        self.audio_x_mask = None
        self.few_shot_mask = None
        self.media_locations = None
        if self.gated_cross_attn_layer_sound is not None:
            self.gated_cross_attn_layer_sound._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return (self.audio_x is not None) and (self.audio_x_mask is not None) and (self.media_locations is not None)

    def condition_audio_x(self, sound_x, sound_x_mask):
        self.sound_x = sound_x
        self.sound_x_mask = sound_x_mask

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer_sound is not None:
            if self.sound_x is None:
                raise ValueError("sound_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer_sound(
                lang_x,
                self.sound_x,
                self.sound_x_mask,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )
        
        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        audio_hidden_size,
        max_window_per_audio,
        cross_attn_every_n_layers,
        gradient_checkpointing,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        self.gated_cross_attn_layers_sound = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, 
                    dim_audio=audio_hidden_size,
                    max_window_per_audio=max_window_per_audio, 
                    only_attend_immediate_media=False,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )

        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_audio_x = False

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layers_sound, decoder_layer, gradient_checkpointing
                    )
                    for gated_cross_attn_layers_sound, decoder_layer in zip(
                        self.gated_cross_attn_layers_sound, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        media_locations = input_ids == self.media_token_id

        use_cached_media_locations = (
            self._use_cached_audio_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_audio_x(None, None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
