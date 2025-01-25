import torch
from transformers.quantizers import quantizer_bitnet
from transformers.integrations import bitnet
import hashlib
import os

import sys
sys.path.insert(2, '..')

counter = 0

def get_space_usage(tensor):
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    space_in_bytes = num_elements * element_size
    return space_in_bytes

'''
**RSR Preprocess**

Runs the preprocess using the 'RSRTernaryMultiplier' 
and saves the blocks as a tensor in a file.
'''
def preprocess_patch(self, model, *args, **kwargs):
    from transformers.integrations.bitnet import unpack_weights, BitLinear, pack_weights
    from torch_impl.multipliers import RSRTernaryMultiplier

    def _apply_preprocess(model, current_key_name=None):
        global counter
        for name, module in model.named_children():
            if current_key_name is None:
                current_key_name = []
            current_key_name.append(name)

            if isinstance(module, BitLinear):
                counter += 1

                bit_linear = model._modules[name]
                device = bit_linear.weight.device
                # Get the ternary matrix
                matrix_158 = unpack_weights(bit_linear.weight, dtype=bit_linear.dtype).T
                # Store the ternary matrix directly
                bit_linear.register_buffer("ternary_matrix", pack_weights(matrix_158.to(device).contiguous()))
                # Calculate a hash for this BitLinear module
                uid = hashlib.sha256(bit_linear.weight.to(torch.device("cpu")).numpy().tobytes()).hexdigest()[:8]
                
                if os.path.exists(f"../data/tensor_{uid}.pt"):
                    print(f"Loading tensors: {counter} / {224}", end="\r")
                    with open(f"../data/tensor_{uid}.pt", "rb") as f:
                        data = torch.load(f, weights_only=True)
                        bit_linear.register_buffer("rsr_matrix", pack_weights(data.to(device)))
                else:
                    print(f"Building tensors: {counter} / {224}", end="\r")
                    multiplier = RSRTernaryMultiplier(
                        A=matrix_158, 
                        k=2 # all matrices are within [2^12, 2^13]
                    )
                    data = multiplier.get_agg_matrix()
                    bit_linear.register_buffer("rsr_matrix", data.to(device))
                    with open(f"../data/tensor_{uid}.pt", "wb") as f:
                        torch.save(data, f)

                del bit_linear.weight
                del matrix_158

            if len(list(module.children())) > 0:
                _apply_preprocess(module, current_key_name)
            # Remove the last key for recursion
            current_key_name.pop(-1)
    
    _apply_preprocess(model)
    return model

'''
**RSR inference**

During the inference, we either use the 'RSR' method or 'Naive' standard.
'''
def rsr_forward(self, input, method='RSR'):
    from transformers.integrations.bitnet import unpack_weights
    
    input_quant, input_scale = self.activation_quant(input)
    y = (input_quant.to(self.dtype) @ unpack_weights(self.rsr_matrix, dtype=self.dtype)).permute(1, 0, 2).reshape(input.size(1), -1).unsqueeze(0)
    y = self.post_quant_process(y, self.weight_scale, input_scale)
    if self.bias is not None:
        y += self.bias.view(1, -1).expand_as(y)
    return y

def standard_forward(self, input):
    from transformers.integrations.bitnet import unpack_weights
    
    input_quant, input_scale = self.activation_quant(input)
    # To make the comparison fair, we multiply 'contiguous' matrices
    y = input_quant.to(self.dtype) @ unpack_weights(self.ternary_matrix, dtype=self.dtype)
    y = self.post_quant_process(y, self.weight_scale, input_scale)
    if self.bias is not None:
        y += self.bias.view(1, -1).expand_as(y)
    return y


# Applies monkey patches
def apply_patch(method):
    quantizer_bitnet.BitNetHfQuantizer._process_model_after_weight_loading = preprocess_patch
    if method == 'RSR':
        bitnet.BitLinear.forward = rsr_forward
    else:
        bitnet.BitLinear.forward = standard_forward