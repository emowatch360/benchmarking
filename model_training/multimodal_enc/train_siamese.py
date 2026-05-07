# Minimal Siamese network trainer for 1D time-series, using contrastive loss
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .models.siamese_net import SiameseNet
from .train import EarlyStopping
from torch.nn import TripletMarginLoss
from sklearn.neighbors import KNeighborsClassifier

class SiameseTrainer:
    """Trainer for SiameseNet."""

    def __init__(self, cfg, print_log_path: str | None = None):
        self.cfg = cfg
        self.print_log_path = print_log_path

        self.device = cfg['training']['device']
        self.lr = cfg['training']['learning_rate']
        self.epochs = cfg['training']['epochs']
        self.use_scheduler = cfg['training']['use_scheduler']
        self.patience = cfg['early_stopping']['patience']
        self.delta = cfg['early_stopping']['delta']
        # self.num_class = cfg['model']['output_dim']
        self.label_type = cfg['data']['label_type'] # discrete or continuous
        assert self.label_type in ['discrete'], "label_type must be 'discrete'!"
        self.margin = cfg['training']['siamese_margin']
        self.p = cfg['training']['siamese_p']  # p-norm for distance metric
        self.knn_k = cfg['training']['siamese_knn_k']
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.knn_k, p=self.p)
        
        self.model = SiameseNet(
            physio_input_dim=cfg['model']['physio_input_dim'],
            sleep_input_dim=cfg['model']['sleep_input_dim'],
            demo_input_dim=cfg['model']['demo_input_dim'],
            proj_head_dim=cfg['model']['proj_head_dim'],
            conv_layers=cfg['model']['conv_layers'],
            sp_conv_layers=cfg['model']['sp_conv_layers'],
            fcnn_layers=cfg['model']['fcnn_layers'],
            dropout=cfg['training']['dropout'],
            output_dim=cfg['model']['output_dim'],
            feature_split=cfg['model']['feature_split'],
            model_type=cfg['model']['model_type']
        )

        self.chosen_train_loss = float('inf')
        self.chosen_val_loss = float('inf')
        self.chosen_epochs = -1

    def train(self, train_loader, val_loader=None):
        self.model.to(self.device)
        criterion = TripletMarginLoss(margin=self.margin, p=self.p, reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                          verbose=True, threshold=1e-4, min_lr=1e-6)

        early_stopper = EarlyStopping(patience=self.patience, delta=self.delta,
                                      restore_best=True)

        train_valid_metrics = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_losses = []
            # initalize an array to save embedding vectors
            for x1_batch, x2_batch, x3_batch, y_batch, group_labels in train_loader:
                skip_batch = False
                x1_batch, x2_batch, x3_batch, y_batch, group_labels = x1_batch.to(self.device), x2_batch.to(self.device), x3_batch.to(self.device), y_batch.to(self.device), group_labels.to(self.device)
                optimizer.zero_grad()
                embedding_mat = self.model.forward_once(x1_batch, x2_batch, x3_batch)
                # for each sample, find a positive and negative sample within the batch
                positive_idx_list = []
                negative_idx_list = []
                for idx in range(embedding_mat.size(0)): # iterate through each sample in a batch
                    anchor_label = y_batch[idx]
                    positive_indices = (y_batch == anchor_label).nonzero(as_tuple=True)[0]
                    negative_indices = (y_batch != anchor_label).nonzero(as_tuple=True)[0]
                    if len(positive_indices) > 1 and len(negative_indices) > 0:
                        positive_indices = positive_indices[positive_indices != idx]
                        # randomly select a postiive index different from idx
                        positive_idx = positive_indices[torch.randint(high=len(positive_indices), size=(1,)).item()].item()
                        negative_idx = negative_indices[torch.randint(high=len(negative_indices), size=(1,)).item()].item()
                        positive_idx_list.append(positive_idx)
                        negative_idx_list.append(negative_idx)
                    else:
                        skip_batch = True
                        with open(self.print_log_path, 'a') as log_file:
                            log_file.write("Skipping batch during training due to insufficient positive or negative samples.\n")
                # compute triplet loss per batch
                if skip_batch:
                    continue
                positive_idx_tensor = torch.tensor(positive_idx_list, device=embedding_mat.device, dtype=torch.long)
                negative_idx_tensor = torch.tensor(negative_idx_list, device=embedding_mat.device, dtype=torch.long)
                positive_mat = embedding_mat[positive_idx_tensor]
                negative_mat = embedding_mat[negative_idx_tensor]
                mean_triplet_loss = criterion(anchor=embedding_mat, positive=positive_mat, negative=negative_mat) # returns mean by default
                epoch_losses.append(mean_triplet_loss.item())
                mean_triplet_loss.backward()
                optimizer.step()
            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('inf')

            if val_loader is not None:
                avg_val_loss = self.evaluate_triplet_loss(val_loader)
                val_acc = None
            else:
                avg_val_loss, val_acc = avg_train_loss, None

            if self.use_scheduler and val_loader is not None:
                scheduler.step(avg_val_loss)

            if self.print_log_path is not None:
                with open(self.print_log_path, 'a') as f:
                    f.write(
                        f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | "
                        f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc if val_acc is not None else 'NA'}\n"
                    )

            train_valid_metrics.append(
                {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                }
            )

            early_stopper(avg_train_loss, avg_val_loss, epoch, self.model)
            if early_stopper.early_stop:
                if early_stopper.restore_best and early_stopper.best_model_state is not None:
                    self.model.load_state_dict(early_stopper.best_model_state)
                self.chosen_train_loss, self.chosen_val_loss, self.chosen_epochs = (
                    early_stopper.return_current_best_model_param()
                )
                if self.print_log_path is not None:
                    with open(self.print_log_path, 'a') as f:
                        f.write(
                            f"Early stopping at epoch {epoch}, "
                            f"Chosen epoch {self.chosen_epochs}\n"
                        )
                break
        if self.chosen_epochs == -1:
            self.chosen_train_loss, self.chosen_val_loss, self.chosen_epochs = (
                early_stopper.return_current_best_model_param()
            )
            if early_stopper.restore_best and early_stopper.best_model_state is not None:
                self.model.load_state_dict(early_stopper.best_model_state)
        self.fit_knn_classifier(train_loader)
        return train_valid_metrics
    
    def fit_knn_classifier(self, train_loader):
        # once the model is trained, call this function to save all embeddings along with their labels for kNN
        self.model.eval()
        self.db_embedding_list = []
        self.db_label_list = []
        with torch.no_grad():
            for x1_batch, x2_batch, x3_batch, y_batch, _ in train_loader:
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)
                x3_batch = x3_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                embedding_mat = self.model.forward_once(x1_batch, x2_batch, x3_batch)
                self.db_embedding_list.append(embedding_mat.cpu().numpy())
                self.db_label_list.append(y_batch.cpu().numpy())
        self.db_embedding_list = np.concatenate(self.db_embedding_list, axis=0)
        self.db_label_list = np.concatenate(self.db_label_list, axis=0)
        self.knn_classifier.fit(self.db_embedding_list, self.db_label_list)

    def evaluate_triplet_loss(self, dataloader, verbose: bool = True):
        was_training = self.model.training
        self.model.eval()
        criterion = TripletMarginLoss(margin=self.margin, p=self.p, reduction='sum')
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for x1_batch, x2_batch, x3_batch, y_batch, _ in dataloader:
                skip_batch = False
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)
                x3_batch = x3_batch.to(self.device)
                y_batch  = y_batch.to(self.device)
                embedding_mat = self.model.forward_once(x1_batch, x2_batch, x3_batch).cpu()
                # for each sample, find a positive and negative sample within the batch
                positive_idx_list = []
                negative_idx_list = []
                for idx in range(embedding_mat.size(0)):
                    anchor_label = y_batch[idx]
                    positive_indices = (y_batch == anchor_label).nonzero(as_tuple=True)[0]
                    negative_indices = (y_batch != anchor_label).nonzero(as_tuple=True)[0]
                    if len(positive_indices) > 1 and len(negative_indices) > 0:
                        positive_indices = positive_indices[positive_indices != idx]
                        # randomly select a postiive index different from idx
                        positive_idx = positive_indices[torch.randint(high=len(positive_indices), size=(1,)).item()].item()
                        negative_idx = negative_indices[torch.randint(high=len(negative_indices), size=(1,)).item()].item()
                        positive_idx_list.append(positive_idx)
                        negative_idx_list.append(negative_idx)
                    else:
                        skip_batch = True
                        with open(self.print_log_path, 'a') as log_file:
                            log_file.write("Skipping batch during evaluation due to insufficient positive or negative samples.\n")
                # compute triplet loss
                if skip_batch:
                    continue
                positive_idx_tensor = torch.tensor(positive_idx_list, device=embedding_mat.device, dtype=torch.long)
                negative_idx_tensor = torch.tensor(negative_idx_list, device=embedding_mat.device, dtype=torch.long)
                positive_mat = embedding_mat[positive_idx_tensor]
                negative_mat = embedding_mat[negative_idx_tensor]
                triplet_loss_sum = criterion(anchor=embedding_mat, positive=positive_mat, negative=negative_mat)
                total_loss += triplet_loss_sum.item()
                bs = y_batch.size(0)
                total_examples += bs
        avg_loss = total_loss / max(1, total_examples)
        if was_training:
            self.model.train()
        if verbose:
            with open(self.print_log_path, 'a') as log_file:
                log_file.write(f"Loss: {avg_loss:.4f}\n")
        return avg_loss
    
    def predict_labels(self, dataloader):
        was_training = self.model.training
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for x1_batch, x2_batch, x3_batch, y_batch, _ in dataloader:
                x1_batch = x1_batch.to(self.device)
                x2_batch = x2_batch.to(self.device)
                x3_batch = x3_batch.to(self.device)
                y_batch  = y_batch.to(self.device)
                embedding_mat = self.model.forward_once(x1_batch, x2_batch, x3_batch).cpu().numpy()
                predict_prob = self.knn_classifier.predict_proba(embedding_mat)
                preds = np.argmax(predict_prob, axis=1)
                all_preds.extend(preds)
                all_probs.extend(predict_prob)
                all_targets.extend(y_batch.cpu().numpy())
        if was_training:
            self.model.train()
        return np.array(all_preds), np.array(all_probs), np.array(all_targets)

    def return_model_param_dict(self):
        return {
            'train-loss': self.chosen_train_loss,
            'val-loss': self.chosen_val_loss,
            'num-epochs': self.chosen_epochs,
        }

    def save_model(self, model_save_path: str):
        torch.save(self.model.state_dict(), model_save_path)

    def load_model(self, model_load_path: str, map_location=None):
        state_dict = torch.load(model_load_path, map_location=map_location or self.device)
        self.model.load_state_dict(state_dict)
        return self.model
