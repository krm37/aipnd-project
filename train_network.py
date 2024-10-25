import torch
from torch import nn

def train_network(model:nn.Module,
          trainloader: torch.utils.data.DataLoader,
          validloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          epochs: int,
          print_every: int,
          device: str
          ):
    model.to(device)

    steps = 0
    running_loss = 0
    for epoch in range(epochs):

        for images, labels in trainloader:
            model.train() 
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            logits = model.forward(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                
                with torch.no_grad():
                    
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                    
                        logits = model.forward(images)
                        loss = criterion(logits, labels)
                        valid_loss+=loss.item()
                    
                        #calculation steps    
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                        
                    
                    train_loss=running_loss/print_every
                    valid_loss=valid_loss/len(validloader)
                    accuracy=accuracy/len(validloader)
                    
                
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {train_loss:.3f}.. "
                        f"Valid loss: {valid_loss:.3f}.. "
                        f"Valid accuracy: {accuracy:.3f}")

                    running_loss = 0
                

    else:   
        return model,criterion,optimizer.state_dict(),epochs


