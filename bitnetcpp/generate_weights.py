import numpy as np

# matrix * vector
# vector.size = K
# matrix.size = M * K
M = 2048
K = 2048

ternary_weights = np.array(np.random.randint(-1, 2, M * K)).astype(np.float32)

ternary_weights = ternary_weights.reshape(M, K)
origin_weight = ternary_weights.astype(np.float32)

print(origin_weight.shape)

origin_weight.tofile("build/original_weight_matrix.bin")