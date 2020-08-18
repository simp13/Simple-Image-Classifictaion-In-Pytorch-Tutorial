from tqdm import tqdm
import torch 
import config 


def metrics_batch(output, target):
    # get output class
    pred = output.argmax(dim=1, keepdim=True)
    
    # compare output class with target class
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def train_fn(model,data_loader,optimizer):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader,total=len(data_loader))
    for data in tk:
        for k,v in data.items():
            data[k] = v.to(config.DEVICE)
        
        optimizer.zero_grad()
        _,loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    
    return fin_loss / len(data_loader)


def eval_fn(model,data_loader):
    model.eval()
    fin_loss = 0.0
    fin_accuracy = 0.0
    tk = tqdm(data_loader,total=len(data_loader))
    with torch.no_grad():
        for data in tk:
            for k,v in data.items():
                data[k] = v.to(config.DEVICE)
            
            batch_preds,loss = model(**data)
            fin_loss += loss.item()
            fin_accuracy += metrics_batch(batch_preds,data['targets'])
        
        return fin_accuracy / len(data_loader),fin_loss /len(data_loader) 