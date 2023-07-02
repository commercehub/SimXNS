from transformers import AutoModel
import torch

model = torch.load("/home/ec2-user/cadeera-datasets/master_models/initial_scrape_test/checkpoint-42000/pytorch_model.bin")
state_dict = {}
for k, v in model.items():
    if k.startswith('lm'):
        state_dict[k.replace("lm.","")] = v.to(torch.float32)
    
print(state_dict)

model = AutoModel.from_pretrained("microsoft/deberta-v3-large", state_dict=state_dict)
model.save_pretrained("./model/", from_pt=True)
print(model)

model2 = AutoModel.from_pretrained("./model/", state_dict=state_dict)
print(model2)