from transformers.quantizers import quantizer_bitnet

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
    quantizer_bitnet.BitNetHfQuantizer._process_model_after_weight_loading = preprocess_after_loading