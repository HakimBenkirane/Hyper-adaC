import torch
import numpy as np
from .utils import print_network, get_optim, CIndex_lifeline, accuracy_cox, cox_log_rank, EarlyStopping, Monitor_CIndex
from src.graph_model.gcnn_surv import GNN_Surv
from torch_geometric.loader import DataLoader
from .utils import CoxLoss
from torch.optim.lr_scheduler import ExponentialLR



def train_loop_survival(epoch, model, loader, optimizer, l2_reg=0., scheduler=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    
    for batch_idx, (data_wsi, event_time, c) in enumerate(loader):

        data_wsi = data_wsi.to(device)
        event_time = event_time.type(torch.FloatTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        risk_score = model(data_wsi, event_time)
        loss =  CoxLoss(event_time, c, risk_score, device)
        loss_value = loss.item()

        all_risk_scores.append(risk_score.cpu().detach().numpy())
        all_censorships.append(c.cpu().detach().numpy())
        all_event_times.append(event_time.cpu().detach().numpy())

        train_loss_surv += loss_value
        train_loss += loss_value

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss_value))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if scheduler:
        scheduler.step()

    # calculate loss and error for epoch
    all_risk_scores = np.concatenate(all_risk_scores).reshape(1, -1)[0]
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    print(all_risk_scores.shape)
    print(all_event_times.shape)
    print(all_censorships.shape)
    cindex_epoch = CIndex_lifeline(all_risk_scores, all_censorships, all_event_times)
    pvalue_epoch = cox_log_rank(all_risk_scores, all_censorships, all_event_times) 
    surv_acc_epoch = accuracy_cox(all_risk_scores, all_censorships)

    print('Epoch: {}, train_loss_surv: {:.4f}, train_p-value: {:.4f}, train_accuracy: {:4f} ,train_c_index: {:.4f}'.format(epoch, train_loss_surv, pvalue_epoch, surv_acc_epoch, cindex_epoch))






def validation_loop_survival(model, loader, l2_reg):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    print('\n')
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_attn_scores = []
    all_coords = []
    
    for batch_idx, (data_wsi, event_time, c) in enumerate(loader):

        data_wsi = data_wsi.to(device)
        event_time = event_time.type(torch.FloatTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            risk_score = model(data_wsi, event_time, return_attn=False)
        loss =  CoxLoss(event_time, c, risk_score, device)
        loss_value = loss.item()

        all_risk_scores.append(risk_score.cpu().numpy())
        all_censorships.append(c.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())
        #all_attn_scores.append(attn_scores)
        #all_coords.append(coords.cpu().numpy())

        val_loss_surv += loss_value
        val_loss += loss_value

    all_risk_scores = np.concatenate(all_risk_scores).reshape(1, -1)[0]
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    cindex_epoch = CIndex_lifeline(all_risk_scores, all_censorships, all_event_times)
    pvalue_epoch = cox_log_rank(all_risk_scores, all_censorships, all_event_times) 
    surv_acc_epoch = accuracy_cox(all_risk_scores, all_censorships)

    print('Eval: validation_loss_surv: {:.4f}, validation_p-value: {:.4f}, validation_accuracy: {:4f} ,validation_c_index: {:.4f}'.format(val_loss_surv, pvalue_epoch, surv_acc_epoch, cindex_epoch))
    return cindex_epoch, all_risk_scores



def train(datasets, fold, output_dim=128, batch_size=32, n_epochs=10, opt_name='adam', lr=1e-3, reg=1e-2, l2_reg=1e-2, model_type='gnn', early_stopping=False):
    """   
        train for a single fold
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nTraining Fold {}!'.format(fold))

    print('\nInit train/val/test splits...', end=' ')
    train_dataset, val_dataset = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_dataset)))
    print("Validating on {} samples".format(len(val_dataset)))


    print('Done!')
    
    print('\nInit Model...', end=' ')


    if model_type =='gnn':
        model = GNN_Surv(output_dim=output_dim)
    else:
        raise NotImplementedError
    
    model = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, optimizer=opt_name, lr=lr, reg=reg)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(n_epochs):
        train_loop_survival(epoch=epoch, model=model, loader=train_loader, optimizer=optimizer, l2_reg=l2_reg, scheduler=None)
        validation_loop_survival(model=model, loader=val_loader, l2_reg=l2_reg)


    val_cindex, all_risk_scores = validation_loop_survival(model=model, loader=val_loader, l2_reg=l2_reg)
    return val_cindex, all_risk_scores
