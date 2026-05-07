import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, softmax
from .base import BaseDGAlgorithm
from .dro import DRO
import torch.utils.data as data_utils
import torch.nn.functional as F

# class HHISS(BaseDGAlgorithm):
class HHISS(DRO):
    def __init__(self, model, device, eta, domains, penalty_weight=1e4,
                 teacher_trade_off=0.5, print_log_path=None):
        # super().__init__(model, device)
        super().__init__(model, device, eta, domains)
        self.penalty_weight = penalty_weight
        self.teacher_trade_off = teacher_trade_off  # weight for the teacher-student loss
        # teacher model and student model are identical at the start
        self.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)
        self.print_log_path = print_log_path

    def __call__(self, *train_batch, criterion=None):
        x1_batch, x2_batch, x3_batch, _, _ = train_batch

        # DRO remains the base HHISS objective because HHISS inherits from DRO.
        dro_loss, _ = super().__call__(*train_batch, criterion=criterion)

        # continuous labels from teacher model as soft targets for student model
        # model called for the second time (first time within DRO) - can be slightly noisy because of different dropout
        _, student_logits = self.model(x1_batch, x2_batch, x3_batch)
        with torch.no_grad():
            _, teacher_logits = self.teacher_model(x1_batch, x2_batch, x3_batch)
        distillation_loss = soft_cross_entropy(student_logits, teacher_logits)
        loss = dro_loss + self.teacher_trade_off * distillation_loss
        return loss, loss
    
    def prune_model(self, train_loader, optimizer, 
                    # all_data, all_label, all_user, 
                    prune_amount):
        count_parameters(self.model)
        # from train loader get all data 
        all_x1, all_x2, all_x3, all_y, all_groups = [],[],[],[],[]
        for x1, x2, x3, y, groups in train_loader:
            all_x1.extend(x1)
            all_x2.extend(x2)
            all_x3.extend(x3)
            all_y.extend(y)
            all_groups.extend(groups)
        all_x1, all_x2, all_x3 = np.array(all_x1), np.array(all_x2), np.array(all_x3)
        all_y, all_groups = np.array(all_y), np.array(all_groups)
        # elif prune_mechanism in ['HHISS','ourIRM','ourDRO','ourVrex']:
        unique_groups = np.unique(all_groups)
        all_masks=[]
        for group in unique_groups:
            group_idx=np.argwhere(np.isin(all_groups, [group])).ravel() # ravel returns a flattened 1D array
            group_x1 = all_x1[group_idx]
            group_x2 = all_x2[group_idx]
            group_x3 = all_x3[group_idx]
            group_y = all_y[group_idx]
            current_group = all_groups[group_idx]
            group_train_set = PruneDataset(group_x1, group_x2, group_x3, group_y, current_group)
            group_data_loader = data_utils.DataLoader(group_train_set, batch_size=32, shuffle=True)
            model_mask, _ = gradient_prune_unstruct1(optimizer, self.model, group_data_loader,
                                                        prune_amount, loss_calculator=self, device = self.device)
            all_masks.append(model_mask)
        model_mask=all_masks[0]
        for mask in all_masks:
            for i,layer in enumerate(mask):
                model_mask[i]=model_mask[i].int()&layer.int()
        self.model = apply_mask(self.model, model_mask, print_log_path=self.print_log_path) #, struct_mask)
        return self.model, model_mask

# Helper function for soft cross entropy loss
def soft_cross_entropy(logits, teacher_logits):
    """
    Computes the soft cross entropy loss between logits and target probability distributions.
    logits: Tensor of shape (batch_size, num_classes)
    target_probs: Tensor of shape (batch_size, num_classes)
    Returns: scalar loss
    """
    log_probs = log_softmax(logits, dim=1)
    target_probs = softmax(teacher_logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1).mean()
    return loss

""" Prune Dataset """
class PruneDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, x3, y, group):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y = y
        self.group = group
        
    def __len__(self):
        return len(self.group)

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.x3[index], self.y[index], self.group[index]
        
def count_parameters(model):
    """
    computes the fraction of non-zero parameters
    """
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    for i in layer_names:
        #print(i)
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
            non_zero_parameters += np.count_nonzero(weights)
    return non_zero_parameters/total_weights

def calculate_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.sum(param ** 2)
    return torch.sqrt(l2_norm)
    
def calculate_l1_norm(model):
    l1_norm = 0.0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm

def gradient_prune_unstruct1(optimizer, model, loader, prune_amount, loss_calculator, device = "cuda"):
    importance_score = []
    batch = 0
    for batch_idx, (x1, x2, x3, targets, user) in enumerate(loader):
        batch += 1
        optimizer.zero_grad()
        x1, x2, x3, targets, user= x1.to(device), x2.to(device), x3.to(device), targets.to(device), user.to(device)
        # outputs = model(x1, x2, x3)
        # outputs = outputs.type(torch.FloatTensor).to(device)
        loss, _ = loss_calculator(x1, x2, x3, targets, user) # replace loss_calculator with HHISS object
        #loss += 0.0001 * calculate_l1_norm(model)
        loss += 0.0005 * calculate_l2_norm(model)
        loss.backward()
        batch_score = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                b_score = module.weight.grad.detach().clone()
                batch_score.append(b_score)
        if batch == 1:
            importance_score = batch_score
        else:
            sum_is = []
            for i,b in zip (importance_score,batch_score):
                sum_is.append(i+b)
            importance_score = sum_is
    #print(importance_score)
    i=0       
    for module in model.modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            importance_score[i] = torch.abs(module.weight * importance_score[i])
            #importance_score[i] = torch.abs(importance_score[i])
            i += 1
    #print(importance_score)
    fine_mask = []
    score = []
    i = 0
    for imp_score in importance_score:
        if i == 0: score = torch.reshape(imp_score.data, (-1,))
        else : score = torch.cat((score,torch.reshape(imp_score.data, (-1,))),0)
        i += 1
    split_val = torch.quantile(score,prune_amount)
    for imp_score in importance_score:
        fine_mask.append(torch.where(imp_score <= split_val, 0.0,1.0))    
    """
    for m in fine_mask:
        print(m.size())
    """
    count_parameters(model)
    # check that mask and modules are aligned (ignoring the last linear layer)
    for m, mask in zip(
        [m for m in model.modules()
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))
        and not (isinstance(m, nn.Linear) and m.out_features == 2)],
        fine_mask[:-1],     # drop last mask
        ):
        # print(m.weight.shape, mask.shape)
        assert(np.all(m.weight.shape == mask.shape))

    return fine_mask, importance_score

def apply_mask(model, model_mask, print_log_path):
    i = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)) and not (
                isinstance(module, nn.Linear) and module.out_features == 2 # currently assumes binary classification 
            ):
                mask = model_mask[i].to(module.weight.device)
                assert np.all(mask.shape == module.weight.shape)
                module.weight.mul_(mask)
                i += 1
    with open(print_log_path, 'a') as log_file:
        log_file.write(f'No. of non-zero parameters after pruning: {count_parameters(model)}\n')
    return model
    
   
    

    
