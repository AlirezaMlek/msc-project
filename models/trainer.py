import os.path
import torch
import pickle


"""
    train path
"""
def train(model, train_loader, loss_fn, optimizer, valid_loader, validator):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    last_step_file = f'./cache/{model.name}/last_step.pickle'
    if os.path.exists(last_step_file):
        with open(last_step_file, 'rb') as file:
            last_step = pickle.load(file)
    else:
        last_step = 0

    total_steps = len(train_loader)

    for epoch in range(3):
        for step, batch in enumerate(train_loader):
            if step < last_step: continue

            inputs = batch[0]
            label = batch[1]
            outputs = model(inputs)
            # attention_mask = torch.squeeze(batch['attention_mask'], dim=1)

            # outputs = select_masked(outputs, attention_mask)

            # label = {'input_ids': batch['labels'], 'attention_mask': batch['attention_mask'],
            #          'token_type_ids': batch['token_type_ids']}
            # label = model.forward_label(label)
            # label = select_masked(label, attention_mask)



            loss = torch.tensor(0.0)
            for l, o in zip(label, outputs):
                # l = torch.nn.Softmax(dim=0)(l)
                o = torch.nn.Softmax(dim=0)(o)
                loss += loss_fn(o, l)

            loss /= len(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                print(f'Epoch {epoch}, Step {step}/{total_steps}, Loss {loss.item()}')
                model.save_fcs()
                with open(last_step_file, 'wb') as f:
                    pickle.dump(step, f)


        validator(model, valid_loader)



"""
    convert redundant outputs to zero by using mask
"""
def select_masked(x, mask):
    a, b = mask.shape

    tmp = [torch.tensor([]) for _ in range(a)]
    for i in range(a):
        for j in range(b):
            if mask[i][j] != 0:
                tmp[i] = torch.cat((tmp[i],x[i,j]), dim=0)
    return tmp
