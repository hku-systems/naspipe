import json
import numpy as np
import sys

# encoder_ops = 92
# decoder_ops = 96
# encoder_layers = 24
# decoder_layers = 24

encoder_ops = 48
decoder_ops = 48
encoder_layers = 16
decoder_layers = 16

subnet = []
p = [0, 17, 33, 50, 51]
for i in range(320000):
    e_ops = np.random.randint(encoder_ops, size=encoder_layers)
    d_ops = np.random.randint(decoder_ops, size=decoder_layers)
    ops = e_ops.tolist() + d_ops.tolist()
    subnet.append(json.dumps({"op": ops, "part": p}))
json.dump(subnet, open('c_48.json', 'w'), indent=4)