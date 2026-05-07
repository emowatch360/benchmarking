# train.py
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax, log_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .algorithms.irm import IRM
from .algorithms.dro import DRO
from .algorithms.vrex import VREX
from .algorithms.dann import DANN
from .algorithms.base import BaseDGAlgorithm
from .algorithms.hhiss import HHISS
from .models.multimodal_encoder import MultimodalEncoderNet
from .models.dann_adversarial_net import DomainAdversarialNet

from .models.siamese_net import SiameseNet
from ..utils.common_metrics import compute_roc

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, restore_best=True):
        self.patience = patience
        self.delta = delta
        self.restore_best = restore_best
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_epochs = -1
        self.counter = 0
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, train_loss, val_loss, epoch_num, model):
        if val_loss < self.best_val_loss - self.delta:
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.best_epochs = epoch_num
            self.counter = 0
            if self.restore_best:
                # self.best_model_state = model.state_dict()
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def return_current_best_model_param(self):
        return self.best_train_loss, self.best_val_loss, self.best_epochs

class DGMultimodalTrainer:
    def __init__(self, cfg, pretrained_cp_encoder=[], pretrained_sp_encoder=[], pretrained_cp_aggregator=None,
                 pretrained_sp_aggregator=None, teacher_model_path=None, print_log_path=None):
        self.print_log_path = print_log_path
        self.cfg = cfg
        self.cv_type = cfg['data']['cv']
        self.dg_algorithm = cfg['training']['algorithm'] # 'ERM', 'IRM', 'DRO', 'VREX', 'DANN', 'HHISS', 'Siamese'
        if self.dg_algorithm.upper() != 'ERM':
            assert self.cv_type in ['loso', 'groupkfold'], "DGMultimodalTrainer only supports LOSO/GroupKFold cross-validation."
        self.dg_model = cfg['model']['model_type'] # 'cnn', 'resnet', 'mha'
        self.device = cfg['training']['device']
        self.pretrained_cp_encoder = pretrained_cp_encoder
        self.pretrained_sp_encoder = pretrained_sp_encoder
        self.pretrained_cp_aggregator = pretrained_cp_aggregator
        self.pretrained_sp_aggregator = pretrained_sp_aggregator
        self.freeze_then_finetune = cfg['training'].get('freeze_then_finetune', True)
        self.freeze_then_finetune_epochs = cfg['training'].get('freeze_then_finetune_epochs', 20)
        # learning rate and early stopping params
        self.lr=cfg['training']['learning_rate']
        self.epochs=cfg['training']['epochs']
        self.use_scheduler=cfg['training']['use_scheduler']
        self.patience=cfg['early_stopping']['patience']
        self.delta=cfg['early_stopping']['delta']
        self.label_type = cfg['data']['label_type'] # discrete or continuous
        assert self.label_type in ['discrete', 'continuous'], "label_type must be either 'discrete' or 'continuous'"
        # model
        assert self.dg_model.upper() in ['CNN', 'RESNET', 'MHA'], "Unsupported model type!"
        assert self.dg_algorithm.upper() != 'SIAMESE', 'Siamese network requires a different trainer!'
        if self.dg_algorithm.upper() == 'DANN':
            self.model = DomainAdversarialNet(physio_input_dim=cfg['model']['physio_input_dim'],
                                            sleep_input_dim=cfg['model']['sleep_input_dim'],
                                            demo_input_dim=cfg['model']['demo_input_dim'],
                                            proj_head_dim=cfg['model']['proj_head_dim'],
                                            conv_layers=cfg['model']['conv_layers'],
                                            sp_conv_layers=cfg['model']['sp_conv_layers'],
                                            fcnn_layers=cfg['model']['fcnn_layers'],
                                            dropout=cfg['training']['dropout'],
                                            num_domains=len(cfg['data']['users']),
                                            output_dim=cfg['model']['output_dim'],
                                            feature_split=cfg['model']['feature_split'],
                                            model_type=self.dg_model.upper(),
                                            pretrained_cp_encoder=self.pretrained_cp_encoder,
                                            pretrained_sp_encoder=self.pretrained_sp_encoder,
                                            pretrained_cp_aggregator=self.pretrained_cp_aggregator,
                                            pretrained_sp_aggregator=self.pretrained_sp_aggregator)
        else:
            self.model = MultimodalEncoderNet(physio_input_dim=cfg['model']['physio_input_dim'],
                                            sleep_input_dim=cfg['model']['sleep_input_dim'],
                                            demo_input_dim=cfg['model']['demo_input_dim'],
                                            proj_head_dim=cfg['model']['proj_head_dim'],
                                            conv_layers=cfg['model']['conv_layers'],
                                            sp_conv_layers=cfg['model']['sp_conv_layers'],
                                            fcnn_layers=cfg['model']['fcnn_layers'],
                                            output_dim=cfg['model']['output_dim'],
                                            dropout=cfg['training']['dropout'],
                                            feature_split=cfg['model']['feature_split'],
                                            model_type=self.dg_model.upper(),
                                            pretrained_cp_encoder=self.pretrained_cp_encoder,
                                            pretrained_sp_encoder=self.pretrained_sp_encoder,
                                            pretrained_cp_aggregator=self.pretrained_cp_aggregator,
                                            pretrained_sp_aggregator=self.pretrained_sp_aggregator)
        if self.dg_algorithm.upper() == 'IRM':
            self.dg_objective = IRM(self.model, self.device,
                                    penalty_weight=cfg['training']['irm_penalty_weight'])
        elif self.dg_algorithm.upper() == 'DRO':
            self.dg_objective = DRO(self.model, self.device,
                                    eta=cfg['training']['dro_eta'],
                                    domains=cfg['data']['users']) # all unique users
        elif self.dg_algorithm.upper() == 'VREX':
            self.dg_objective = VREX(self.model, self.device,
                                     vrex_lambda=cfg['training']['vrex_lambda'])
        elif self.dg_algorithm.upper() == 'DANN':
            assert self.label_type == 'discrete', 'Group prediction does not support continuous labels!'
            self.dg_objective = DANN(self.model, self.device,
                                     dann_lambda=cfg['training']['dann_lambda'],
                                     domains=cfg['data']['users'])
        elif self.dg_algorithm.upper() == 'ERM': # default
            self.dg_objective = BaseDGAlgorithm(self.model, self.device)
        elif self.dg_algorithm.upper() == 'HHISS':
            state_dict = self.load_model(teacher_model_path)
            self.model.load_state_dict(state_dict)  # replicate the teacher model for training
            hhiss_base_algorithm = cfg['training']['hhiss_base_algorithm'].upper()
            if hhiss_base_algorithm == 'ERM':
                assert HHISS.__mro__[1] == BaseDGAlgorithm, 'incorrect inheritance!' # checking if the parent class is correct
            elif hhiss_base_algorithm == 'DRO':
                assert HHISS.__mro__[1] == DRO, 'incorrect inheritance!'
            else:
                raise ValueError('Incorrect base algorithm for HHISS!')
            self.dg_objective = HHISS(self.model, self.device,
                                      penalty_weight=cfg['training']['irm_penalty_weight'],
                                      teacher_trade_off=cfg['training']['hhiss_teacher_tradeoff'],
                                      eta=cfg['training']['dro_eta'],
                                      domains=cfg['data']['users'],
                                      print_log_path=self.print_log_path)
            self.roc_threshold = cfg['training']['hhiss_roc_threshold']
            self.prune_threshold = cfg['training']['hhiss_prune_threshold']
        else:
            raise ValueError(f"Unsupported algorithm: {self.dg_algorithm}")
        self.chosen_train_loss = float('inf')
        self.chosen_val_loss = float('inf')
        self.chosen_epochs = -1
    
    def set_encoder_frozen(self, freeze_encoders: bool):
    
        encoders = list(self.model.pretrained_cp_encoder) + list(self.model.pretrained_sp_encoder)
        for encoder in encoders:
            for param in encoder.parameters():
                param.requires_grad_(not freeze_encoders)
            if freeze_encoders:
                encoder.eval()
        
        # Guard aggregators
        for aggregator in [self.model.pretrained_cp_aggregator, self.model.pretrained_sp_aggregator]:
            if aggregator is not None:
                for param in aggregator.parameters():
                    param.requires_grad_(not freeze_encoders)
                if freeze_encoders:
                    aggregator.eval()

    def train(self, train_loader, val_loader, class_weights=None):
        self.model.to(self.device)
        has_pretrained = (len(self.pretrained_cp_encoder) + len(self.pretrained_sp_encoder)) > 0

        # Freeze pretrained encoders before optimizer is initialised
        # so frozen params are excluded from the default param group
        if self.freeze_then_finetune and has_pretrained:
            self.set_encoder_frozen(freeze_encoders=True)

        if self.label_type == 'discrete':
            criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
        else:
            criterion = soft_cross_entropy
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=1e-4
        )
        # Scheduler: Reduce LR when a metric (val_loss) has stopped improving
        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                        threshold=1e-4, min_lr=1e-6)
        early_stopper = EarlyStopping(patience=self.patience, delta=self.delta,
                                      restore_best=True)
        train_valid_metrics = []
        for epoch in range(1, self.epochs+1):
            
            # Unfreeze pretrained encoders after freeze_then_finetune_epochs
            if self.freeze_then_finetune and has_pretrained:
                if epoch == self.freeze_then_finetune_epochs + 1:
                    self.set_encoder_frozen(freeze_encoders=False)
                    encoder_params = []
                    for enc in list(self.model.pretrained_cp_encoder) + list(self.model.pretrained_sp_encoder):
                        encoder_params += list(enc.parameters())
                    if self.model.pretrained_cp_aggregator is not None:
                        encoder_params += list(self.model.pretrained_cp_aggregator.parameters())
                    if self.model.pretrained_sp_aggregator is not None:
                        encoder_params += list(self.model.pretrained_sp_aggregator.parameters())
                    optimizer.add_param_group({
                        'params': encoder_params,
                        'lr': self.lr # Or potentially lower LR to self.lr * 0.1 to avoid overwriting pretrained representations?
                    })
                    with open(self.print_log_path, 'a') as log_file:
                        log_file.write(f"Unfreezing pretrained encoders at epoch {epoch}, encoder lr={self.lr:.6f}\n")

            self.model.train() # activates dropout layers, batch normalization

            # Re-enforce eval on frozen encoders after model.train() recursive call
            if self.freeze_then_finetune and has_pretrained and epoch <= self.freeze_then_finetune_epochs:
                for enc in list(self.model.pretrained_cp_encoder) + list(self.model.pretrained_sp_encoder):
                    enc.eval()
                if self.model.pretrained_cp_aggregator is not None:
                    self.model.pretrained_cp_aggregator.eval()
                if self.model.pretrained_sp_aggregator is not None:
                    self.model.pretrained_sp_aggregator.eval()
            
            # This is for x1 = Physio data, x2 = Sleep data, x3 = Non-temporal data.
            alg_loss_list = []
            for x1_batch, x2_batch, x3_batch, y_batch, group_labels in train_loader:
                x1_batch, x2_batch, x3_batch, y_batch, group_labels = x1_batch.to(self.device), x2_batch.to(self.device), \
                                                                    x3_batch.to(self.device), y_batch.to(self.device), group_labels.to(self.device)
                optimizer.zero_grad()
                loss, alg_specific_loss = self.dg_objective(*(x1_batch, x2_batch, x3_batch, y_batch, group_labels),
                                         criterion = criterion)
                loss.backward() # compute gradients
                optimizer.step() # updates the parameters
                alg_loss_list.append(alg_specific_loss.item())
            avg_train_loss, train_roc = self.evaluate(train_loader, verbose=False)
            avg_alg_loss = np.mean(alg_loss_list)
            # Evaluate on validation set
            avg_val_loss, val_roc = self.evaluate(val_loader, verbose=False)
            train_roc_dict = self.evaluate_per_group(train_loader)
            if self.use_scheduler:
                scheduler.step(avg_val_loss)
                current_lr = scheduler.get_last_lr()[0]
                with open(self.print_log_path, 'a') as log_file:
                    log_file.write(f"Current LR: {current_lr:.8f}\n")
            with open(self.print_log_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Alg loss = {avg_alg_loss:.4f}, ROC = {train_roc:.4f} | Val Loss = {avg_val_loss:.4f}, ROC = {val_roc:.4f}\n")
                log_file.write(f'{sorted(train_roc_dict.items(), key=lambda x:x[1], reverse=True)}\n')
            train_valid_metrics.append({'train_loss': avg_train_loss,
                                        'alg_loss': avg_alg_loss,
                                        'train_roc': train_roc,
                                        'val_loss': avg_val_loss, 'val_roc': val_roc})
            # Early stopping check (in SSL case, after encoders are unfrozen)
            if not (self.freeze_then_finetune and has_pretrained and epoch <= self.freeze_then_finetune_epochs):
                early_stopper(avg_train_loss, avg_val_loss, epoch, self.model) # needs access to the model to retrieve the best weights
                if early_stopper.early_stop:
                    if early_stopper.restore_best and early_stopper.best_model_state is not None:
                        self.model.load_state_dict(early_stopper.best_model_state)
                    self.chosen_train_loss, self.chosen_val_loss, self.chosen_epochs = early_stopper.return_current_best_model_param()
                    with open(self.print_log_path, 'a') as log_file:
                        log_file.write(f"Early stopping at epoch {epoch}, Chosen epoch {self.chosen_epochs}\n")
                    break
            # HHISS-specific pruning based on ROC threshold
            if self.dg_algorithm.upper() == 'HHISS':
                if train_roc > self.roc_threshold:
                    with open(self.print_log_path, 'a') as log_file:
                        log_file.write('Perform pruning!!')
                    # perform pruning
                    self.dg_objective.prune_model(train_loader, optimizer, self.prune_threshold)
        if self.chosen_epochs == -1:
            self.chosen_train_loss, self.chosen_val_loss, self.chosen_epochs = (
                early_stopper.return_current_best_model_param()
            )
            if early_stopper.restore_best and early_stopper.best_model_state is not None:
                self.model.load_state_dict(early_stopper.best_model_state)
        return train_valid_metrics

    def evaluate(self, dataloader, verbose: bool = True):
        """
        Evaluates the model on `dataloader` in eval mode with no grad.
        Returns (avg_loss, roc) where:
        - avg_loss is sample-weighted average CrossEntropy over all examples
        - roc is computed once over the concatenated logits/targets
        """
        was_training = self.model.training
        self.model.eval()
        # Sum reduction + divide by total examples → unbiased average
        if self.label_type == 'discrete':
            criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            criterion = soft_cross_entropy
        total_loss = 0.0
        total_examples = 0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for x1_batch, x2_batch, x3_batch, y_batch, _ in dataloader:
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)
                x3_batch = x3_batch.to(self.device)
                y_batch  = y_batch.to(self.device)
                if self.dg_algorithm.upper() == 'DANN':
                    _, logits, _ = self.model(x1_batch, x2_batch, x3_batch, lambda_=self.cfg['training']['dann_lambda'])
                else:
                    _, logits = self.model(x1_batch, x2_batch, x3_batch)
                # accumulate loss as a sum, track how many samples we saw
                loss = criterion(logits, y_batch)
                bs = y_batch.size(0)
                total_loss += loss.item()
                total_examples += bs
                # stash for a single ROC computation at the end
                all_logits.append(logits.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
        avg_loss = total_loss / max(1, total_examples)
        # Compute ROC once over all predictions/targets
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        roc = compute_roc(logits_cat, targets_cat)
        if was_training:
            self.model.train()
        if verbose:
            with open(self.print_log_path, 'a') as log_file:
                log_file.write(f"Loss: {avg_loss:.4f} | ROC: {roc:.4f}\n")
        return avg_loss, roc

    def evaluate_per_group(self, dataloader):
        was_training = self.model.training
        self.model.eval()
        all_x1, all_x2, all_x3, all_y, all_groups = [],[],[],[],[]
        for x1, x2, x3, y, groups in dataloader:
            all_x1.extend(x1)
            all_x2.extend(x2)
            all_x3.extend(x3)
            all_y.extend(y)
            all_groups.extend(groups)
        all_x1 = torch.stack(all_x1).to(self.device)
        all_x2 = torch.stack(all_x2).to(self.device)
        all_x3 = torch.stack(all_x3).to(self.device)
        all_y = torch.stack(all_y)
        all_groups = np.array([g.item() if isinstance(g, torch.Tensor) else g for g in all_groups])
        # elif prune_mechanism in ['HHISS','ourIRM','ourDRO','ourVrex']:
        unique_groups = np.unique(all_groups)
        roc_dict = {}
        with torch.no_grad():
            for group in unique_groups:
                group_idx=np.argwhere(np.isin(all_groups, [group])).ravel() # ravel returns a flattened 1D array
                group_x1 = all_x1[group_idx]
                group_x2 = all_x2[group_idx]
                group_x3 = all_x3[group_idx]
                group_y = all_y[group_idx]
                if self.dg_algorithm.upper() == 'DANN':
                    _, logits, _ = self.model(group_x1, group_x2, group_x3, lambda_=self.cfg['training']['dann_lambda'])
                else:
                    _, logits = self.model(group_x1, group_x2, group_x3)
                # check if group_y has only one class
                if len(torch.unique(group_y)) < 2:
                    roc = -1
                else:
                    roc = compute_roc(logits.detach().cpu(), group_y.detach().cpu())
                roc_dict[int(group)] = round(float(roc), 3)
        if was_training:
            self.model.train()
        return roc_dict

    def predict_labels(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []  # To store softmax probabilities
        with torch.no_grad():
            # This is for x1 = Physio data, x2 = Sleep data, x3 = Non-temporal data.
            for x1_batch, x2_batch, x3_batch, y_batch, _ in dataloader:
                x1_batch, x2_batch, x3_batch = x1_batch.to(self.device), x2_batch.to(self.device), x3_batch.to(self.device)
                if self.dg_algorithm.upper() == 'DANN':
                    _, logits, _ = self.model(x1_batch, x2_batch, x3_batch, lambda_=self.cfg['training']['dann_lambda'])
                else:
                    _, logits = self.model(x1_batch, x2_batch, x3_batch)
                probs = softmax(logits, dim=1)  # Convert logits to probabilities
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())  # Store probabilities
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())  # assumes y_batch is already integer class indices
        return np.array(all_preds), np.array(all_probs), np.array(all_targets)   # Return probabilities along with predictions and targets
    
    def return_model_param_dict(self):
        return {'train-loss': self.chosen_train_loss,
                'val-loss': self.chosen_val_loss,
                'num-epochs': self.chosen_epochs}
    
    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)
    
    def save_finetuned_encoders(self, pretraining_folder, data_type, pretraining_type,
                             conv_layers, batch_size, pretrain_epochs, feature_tag='all'):
        """
        Save finetuned pretrained encoders and aggregators to pretraining folder,
        prefixed with 'finetuned_' and tagged with fold index.
        """
        encoders     = list(self.model.pretrained_cp_encoder) if data_type == 'cp' \
                    else list(self.model.pretrained_sp_encoder)
        aggregator   = self.model.pretrained_cp_aggregator if data_type == 'cp' \
                    else self.model.pretrained_sp_aggregator
        model_type   = self.cfg['model']['model_type']
        feature_split = self.cfg['model']['feature_split']
        dropout      = self.cfg['training']['dropout']

        if len(encoders) == 0:
            print(f'No pretrained {data_type} encoders to save.')
            return

        for i, encoder in enumerate(encoders):
            save_path = os.path.join(
                pretraining_folder,
                f'finetuned_encoder_{data_type}_{pretraining_type}_{model_type}{conv_layers}'
                f'_batch{batch_size}_epoch{pretrain_epochs}_feat_{feature_tag}_{i}_pretrained.pt')
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'encoder_config': {
                    'input_features': feature_split[i],
                    'conv_layers': conv_layers,
                    'dropout': dropout,
                    'num_heads': 4 if model_type.upper() == 'MHA' else 0
                }
            }, save_path)
            print(f'Saved finetuned encoder {i} to {save_path}')

        # Save aggregator for all pretraining types (not just vicreg)
        if aggregator is not None:
            embedding_dim = conv_layers[-1]
            agg_save_path = os.path.join(
                pretraining_folder,
                f'finetuned_aggregator_{data_type}_{pretraining_type}_{model_type}{conv_layers}'
                f'_batch{batch_size}_epoch{pretrain_epochs}_feat_{feature_tag}_pretrained.pt')
            torch.save({
                'aggregator_state_dict': aggregator.state_dict(),
                'aggregator_config': {
                    'num_feature_groups': len(feature_split),
                    'embedding_dim': embedding_dim,
                }
            }, agg_save_path)
            print(f'Saved finetuned aggregator to {agg_save_path}')
    
    def load_model(self, teacher_model_path):
        state_dict = torch.load(teacher_model_path,
                                map_location=self.device,
                                weights_only=True)
        return state_dict

# Helper function for soft cross entropy loss
def soft_cross_entropy(logits, target_probs):
    """
    Computes the soft cross entropy loss between logits and target probability distributions.
    logits: Tensor of shape (batch_size, num_classes)
    target_probs: Tensor of shape (batch_size, num_classes)
    Returns: scalar loss
    """
    log_probs = log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1).mean()
    return loss
