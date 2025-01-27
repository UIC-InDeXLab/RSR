import torch
from transformers.quantizers import quantizer_bitnet
from transformers.integrations import bitnet
import hashlib
import os
import time

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
                # counter += 1

                bit_linear = model._modules[name]
                device = bit_linear.weight.device
                # Get the ternary matrix
                ternary_matrix = unpack_weights(bit_linear.weight, dtype=bit_linear.dtype).T
                # Store the ternary matrix directly
                bit_linear.register_buffer("ternary_matrix", pack_weights(ternary_matrix.to(device)))
                del bit_linear.weight
                del ternary_matrix

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
def comparing_forward(self, input, method='RSR'):
    from transformers.integrations.bitnet import unpack_weights
    from torch_impl.multipliers import RSRTernaryMultiplier
    
    global counter
    
    print(f"{counter + 1} / 224")
    counter += 1
    
    input_quant, input_scale = self.activation_quant(input)
    vector = input_quant.to(self.dtype)
    ter = unpack_weights(self.ternary_matrix, dtype=self.dtype)
    
    '''Start RSR'''
    mult = RSRTernaryMultiplier(ter.to(torch.device("cpu")), k=2)
    c = mult.get_agg_matrix().to(ter.device)
    c1 = c[:,:,0]
    c2 = c[:,:,1]
    m1 = c1.T.contiguous()
    m2 = c2.T.contiguous()
    matrices = torch.stack([m1, m2], dim=0)
    matrices_expanded = matrices.unsqueeze(0).expand(4, -1, -1, -1).flatten(0, 1)
    
    t = time.time()
    v = vector.squeeze(0)
    vectors_expanded = v.unsqueeze(1).expand(-1, 2, -1)
    result = torch.bmm(vectors_expanded.flatten(0, 1).unsqueeze(1), matrices_expanded[:2*v.size(0)]).squeeze(1)
    result = result.view(v.size(0), 2, -1)
    y1 = torch.stack((result[:,0,:], result[:,1,:]), dim=2).reshape(v.size(0), -1).unsqueeze(0)
    print(f"RSR time: {time.time() - t}")
    '''End RSR'''
    
    '''Start Old'''
    t = time.time()
    y = vector @ ter
    print(f"Standard time: {time.time() - t}")
    '''End Old'''
    
    if not torch.allclose(y1, y, atol=1e-2, rtol=1e-2):
        print("Error: outputs do not match!")
    
    y = self.post_quant_process(y, self.weight_scale, input_scale)
    if self.bias is not None:
        y += self.bias.view(1, -1).expand_as(y)
    return y

# Applies monkey patches
def apply_patch(method):
    quantizer_bitnet.BitNetHfQuantizer._process_model_after_weight_loading = preprocess_patch
    bitnet.BitLinear.forward = comparing_forward