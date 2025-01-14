import torch
from transformers.quantizers import quantizer_bitnet
from transformers.integrations import bitnet

def init(self, in_features: int, out_features: int, bias: bool, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features // bitnet.VALUES_PER_ITEM, in_features),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                (1),
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype, device=device))
        else:
            self.bias = None
        
        self.register_buffer("permutations", torch.zeros(10))
        self.register_buffer("segments", torch.zero(10))

def preprocess_after_loading(self, model, *args, **kwargs):
    print(f"_process_model_after_weight_loading()")
    from transformers.integrations.bitnet import unpack_weights, BitLinear
    # ---- [Mohsen]
    def _get_matrices(model, current_key_name=None):
        for name, module in model.named_children():
            if current_key_name is None:
                current_key_name = []
            current_key_name.append(name)

            # Check if the current key is not in the `modules_to_not_convert`
            if isinstance(module, BitLinear):
                bitLinear = model._modules[name]
                matrix_158 = unpack_weights(bitLinear.weight, dtype=bitLinear.dtype)
                print(f"Weight Matrix #2: {matrix_158.shape}")

            if len(list(module.children())) > 0:
                _get_matrices(module, current_key_name)
            # Remove the last key for recursion
            current_key_name.pop(-1)
    
    _get_matrices(model)
    # ----
    return model

def patch():
    bitnet.BitLinear.__init__ = init
    quantizer_bitnet.BitNetHfQuantizer._process_model_after_weight_loading = preprocess_after_loading