import torch
# from model.Depthformer.Depthformer import Depthformer
# from model.Depthformer.depthformer_v2 import DepthformerV2
# from model.Depthformer.depthformer_v4 import DepthformerV4
# from model.Depthformer.depthformer_v6 import DepthformerV6
# from model.Depthformer.depthformer_v7 import DepthformerV7
from model.Depthformer.depthformer_v8 import DepthformerV8

model = DepthformerV8.build(opt={"hidden_dim": 256,
                                 "num_heads": 4,
                                 "num_bins": 256,
                                 "num_aux": 256,
                                 "img_size": (352, 704)},
                            min_depth=0.001, max_depth=80.0)

dummy_input = torch.empty(4, 3, 352, 1216)
dummy_output, centers, dummy_attn_weights = model(dummy_input)
print("Output:", dummy_output.shape)
print("Centers:", centers.shape)
for weight in dummy_attn_weights:
    print("Attn weight:", weight.shape)

num_params = model.count_params()
print("#Params:", num_params)

encoder_num_params = 0
for v in model.encoder.parameters():
    encoder_num_params += v.numel()
print("#Encoder Params:", encoder_num_params)
print("#New params:", num_params - encoder_num_params)
