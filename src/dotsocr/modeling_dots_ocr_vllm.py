from functools import cached_property
from typing import Iterable, Literal, Mapping, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from vllm import ModelRegistry
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import Qwen2VLDummyInputsBuilder
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors

from .configuration_dots import DotsVisionConfig
from .configuration_dots import DotsOCRConfig
from .modeling_dots_vision import DotsVisionTransformer


class DotsOCRImagePixelInputs(TypedDict):
    type: Literal["pixel_values", "image_grid_thw"]

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


class DotsOCRImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds", "image_grid_thw"]
    image_embeds: torch.Tensor
    """Supported types:
    - List[`torch.Tensor`]: A list of tensors holding all images' features.
        Each tensor holds an image's features.
    - `torch.Tensor`: A tensor holding all images' features
        (concatenation of all images' feature tensors).

    Tensor shape: `(num_image_features, hidden_size)`
    - `num_image_features` varies based on
        the number and resolution of the images.
    - `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor


DotsOCRImageInputs = Union[DotsOCRImagePixelInputs, DotsOCRImageEmbeddingInputs]


class DotsOCRMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    pass


class DotsOCRDummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(width=target_width, height=target_height, num_images=num_images),
        }


class DotsOCRProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self) -> DotsOCRConfig:
        config = self.ctx.get_hf_config()
        if not config.__class__.__name__ == 'DotsOCRConfig':
            raise TypeError(f"Expected DotsOCRConfig, got {type(config)}")

        if hasattr(config, "vision_config") and isinstance(config.vision_config, dict):
            config.vision_config = DotsVisionConfig(**config.vision_config)
            
        return config

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": 0}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        return {"image": max_image_tokens, "video": 0}

    def get_hf_processor(
        self,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        **kwargs: object,
    ) -> Qwen2VLProcessor:
        self.get_tokenizer().image_token = "<|imgpad|>" # Ensure image token is set
        processor = self.ctx.get_hf_processor(
            Qwen2VLProcessor,
            image_processor=self.get_image_processor(min_pixels=min_pixels, max_pixels=max_pixels, size=size),
            **kwargs,
        )
        processor.image_token = "<|imgpad|>"
        processor.video_token = "<|video_pad|>"
        return processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessor],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config: DotsOCRConfig = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        # NOTE: Frames are padded to be divisible by `temporal_patch_size`
        # https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L294
        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=DotsOCRProcessingInfo,
    dummy_inputs=DotsOCRDummyInputsBuilder,
)
class DotsOCRForCausalLM(nn.Module, SupportsMultiModal):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
    _tp_plan = {}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality in ("image",):
            return "<|img|><|imgpad|><|endofimg|>"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config: DotsOCRConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        if isinstance(self.config.vision_config, dict):
            vision_config = DotsVisionConfig(**self.config.vision_config)
            self.config.vision_config = vision_config
        else:
            vision_config = self.config.vision_config

        self.vision_tower = DotsVisionTransformer(vision_config)
        self.language_model: Qwen2ForCausalLM = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. " f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} "
                    f"(shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[DotsOCRImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. " f"Got type: {type(pixel_values)}")

            return DotsOCRImagePixelInputs(
                type="pixel_values", pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. " f"Got type: {type(image_embeds)}")
            return DotsOCRImageEmbeddingInputs(
                type="image_embeds", image_embeds=image_embeds, image_grid_thw=image_grid_thw
            )

    def vision_forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        from vllm.distributed import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        assert self.vision_tower is not None

        tp_rank = get_tensor_model_parallel_rank()
        tp = get_tensor_model_parallel_world_size()

        image_grid_thw_chunk = image_grid_thw.chunk(tp)
        image_sizes_consum = torch.tensor([i.prod(-1).sum() for i in image_grid_thw_chunk]).cumsum(dim=0)
        merge_size_square = self.vision_tower.config.spatial_merge_size**2
        image_embedding = torch.zeros(
            (
                pixel_values.shape[0] // merge_size_square,
                self.vision_tower.config.hidden_size,
            ),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )

        if tp_rank < len(image_sizes_consum):
            idx_start = 0 if tp_rank == 0 else image_sizes_consum[tp_rank - 1].item()
            idx_end = image_sizes_consum[tp_rank].item()
            pixel_values_part = pixel_values[idx_start:idx_end]
            image_grid_thw_part = image_grid_thw_chunk[tp_rank]
            image_embedding_part = self.vision_tower(pixel_values_part, image_grid_thw_part)
            image_embedding[idx_start // merge_size_square : idx_end // merge_size_square] = image_embedding_part

        group = get_tensor_model_parallel_group().device_group
        torch.distributed.all_reduce(image_embedding, group=group)
        return image_embedding

    def _process_image_input(self, image_input: DotsOCRImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
            image_embeds = self.vision_forward(pixel_values, grid_thw)[
                :, : self.config.hidden_size
            ]

        # Split concatenated embeddings for each image item.
        merge_size = self.vision_tower.config.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
        return modalities

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += vision_embeddings

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.image_token_id, self.config.video_token_id],
            )

        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[DotsOCRImagePixelInputs] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_id,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None and kwargs.get("pixel_values") is not None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is None:
                inputs_embeds = None
            else:
                assert input_ids is not None
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                )
                input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def patch_vllm_chat_placeholder():
    import vllm
    # return when vllm version > 0.9.1
    if not (vllm.__version_tuple__[0]==0 and vllm.__version_tuple__[1] <= 9 and vllm.__version_tuple__[2] <= 1):
        return
    from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

    ori = BaseMultiModalItemTracker._placeholder_str

    def _placeholder_str(self, modality, current_count: int) -> Optional[str]:
        hf_config = self._model_config.hf_config
        model_type = hf_config.model_type
        if modality in ("image",) and model_type in ["dots_ocr"]:
            return "<|img|><|imgpad|><|endofimg|>"
        return ori(self, modality, current_count)

    BaseMultiModalItemTracker._placeholder_str = _placeholder_str

ModelRegistry.register_model(
    "DotsOCRForCausalLM", DotsOCRForCausalLM,
)


patch_vllm_chat_placeholder()