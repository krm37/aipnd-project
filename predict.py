import torch
import helper
import json
from torch import nn
def predict(model, image_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        
        model.eval()
                
        processed_image = helper.process_image(image_path)
        processed_image=torch.tensor(processed_image).view(3,224,224).float()
        logits = model.forward(processed_image.unsqueeze_(0))
        
        
        ps = torch.exp(logits)
        probs, class_indices = ps.topk(topk, dim=1)
        
        
    return probs,class_indices

def predict_result(model,image_path,top_k=5):
    
    probs,class_indices=predict(model,image_path,top_k)
    probs=probs.numpy().squeeze()
    class_indices=class_indices.numpy().squeeze()
    
    class_names=[]
    for i in class_indices:
        key = next(key for key, val in model.class_to_idx.items() if val == i)
        class_names.append((cat_to_name[str(key)]))

    for prob,name in zip(probs,class_names):
        print(f"{name}: {prob:.3f}..")

args=helper.get_prediction_input_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
image_path=args.input
checkpoint=torch.load(args.checkpoint)
topk=args.topk
    
model=helper.model_dict[checkpoint["arch"]]
model.classifier=helper.create_classifier(checkpoint["input_unit"],checkpoint["hidden_units"],checkpoint["output_unit"])
model.load_state_dict(checkpoint["model"])
model.class_to_idx=checkpoint["class_to_idx"]
    
            
predict_result(model,image_path,top_k=topk)         