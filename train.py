import ast
from datetime import datetime
import os
import torchvision
import helper
from train_network import train_network
from torch import nn
import torch
import time

### getting arguments
args=helper.get_train_input_args()
data_dir=args.data_dir
arch=args.arch
learning_rate=args.learning_rate
hidden_units = ast.literal_eval(args.hidden_units)
epochs=args.epochs
save_dir=args.save_dir

trainloader,validloader,class_to_idx=helper.get_data_loader(data_dir)

print(args.gpu)

if torch.cuda.is_available() and args.gpu:
    device="cuda"
else:
    device="cpu"


model=helper.model_dict[arch]


### Creating model classifier

for param in model.parameters():
    param.requires_grad = False

try:
    input_unit = model.classifier.in_features
except:
    input_unit = model.classifier[0].in_features

output_unit=102
model.classifier=helper.create_classifier(input_unit,hidden_units,output_unit)

### training the model
print_every=40
optimizer=torch.optim.SGD(model.classifier.parameters(),lr=learning_rate)
criterion=nn.NLLLoss()

model,criterion,optimizer_state_dict,epochs=train_network(model,trainloader,validloader,optimizer,criterion,epochs,print_every,device)

### saving the network
checkpoint={
    "model":model.state_dict(),
    "criterion":criterion,
    "optimizer_state_dict":optimizer_state_dict,
    "num_epochs":epochs,
    "hidden_layers":args.hidden_units,
    "output_layer":output_unit,
    "arch": arch,
    "input_unit":input_unit,
    "hidden_units":hidden_units,
    "output_unit":output_unit,
    "class_to_idx":class_to_idx
}

run_id = time.strftime("run_\%Y_\%m_\%d-\%H_\%M_\%S")

save_dir = os.path.join(save_dir, arch, run_id)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))
