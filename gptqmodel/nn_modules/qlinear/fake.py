import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import torch_compile

log = setup_logger()

class FakeQuantLinear(PackableQuantLinear):
    """This was copied from a different PackableQuantLinear, so may need some clean up."""
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True # ??
    SUPPORTS_TRAINING = True # ??
    SUPPORTS_AUTO_PADDING = True # ??
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32] # ??
    SUPPORTS_ADAPTERS = [Lora] # ??

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = False,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.FAKE),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.register_buffer(
            "weight",
            torch.empty(
                (self.out_features, self.in_features), dtype=torch.float16
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None


    def post_init(self):
        # Don't call super.
        pass

    def pack(self, linear: nn.Module, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor=None):
        self.weight[:,:] = linear.weight
        if self.bias is not None:
            self.bias[:] = linear.bias
        else:
            assert linear.bias is None

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, FakeQuantLinear):
            raise ValueError(
                "Only models loaded using FakeQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.FAKE."
            )

        if isinstance(module, FakeQuantLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.weight)
            new_module.bias = nn.Parameter(module.bias)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["FakeQuantLinear", "dequantize_model"]
