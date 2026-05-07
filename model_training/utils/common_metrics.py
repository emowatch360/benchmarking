from sklearn.metrics import classification_report, roc_auc_score
import torch
from torch.utils.data import TensorDataset
import numpy as np

def compute_accuracy(logits, labels):
    if labels.dim() == 2:
        labels = torch.argmax(labels, dim=1)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_roc(logits, true_labels, pred_scores='logits'):
    # if true_labels is one-hot encoded or continuous, convert to class indices
    if true_labels.dim() == 2:
        true_labels = torch.argmax(true_labels, dim=1)
    if pred_scores == 'logits':
        pred_prob = torch.softmax(logits, dim=1)  # Assuming binary classification
    elif pred_scores == 'probabilities':
        pred_prob = logits
    true_labels_numpy = true_labels.cpu().numpy()
    pred_prob_numpy = pred_prob.cpu().numpy()
    assert pred_prob_numpy.shape[1] == 2, "Expected pred_prob to have shape (n_samples, 2) for binary classification"
    roc_val = roc_auc_score(true_labels_numpy, pred_prob_numpy[:, 1], average='macro')
    return roc_val

def compute_classification_report(pred_labels, true_labels, labels=None,
                                  target_names=None, verbose=True, print_log_path=None):
    # if true_labels is one-hot encoded or continuous, convert to class indices
    if true_labels.shape[0] != true_labels.size:
        true_labels = np.argmax(true_labels, axis=1)
    # generate detailed classification report as a dict
    report_dict = classification_report(
        true_labels, pred_labels, labels=labels, target_names=target_names, output_dict=True, digits=4, zero_division=0
    )
    # print nicely formatted report
    if verbose and print_log_path is not None:
        with open(print_log_path, 'a') as log_file:
            log_file.write(f'{classification_report(true_labels, pred_labels, labels=labels, target_names=target_names, digits=4, zero_division=0)}\n')
    return report_dict

def compute_best_threshold_for_macro_f1(pred_score_matrix, true_labels, labels, print_log_path=None):
    assert pred_score_matrix.shape[1] == 2, "Best threshold computation is only for binary classification."
    # we perform this only for binary classes, on the validation dataset
    thresholds = np.linspace(0.0, 1.0, num=20, endpoint=False)[1:]  # Exclude 0.0
    # if true_labels contains only one unique value, return default threshold
    if len(np.unique(true_labels)) == 1:
        return 0.5, None
    best_macro_f1 = -1.0
    best_threshold = 0.5
    if print_log_path is not None:
        with open(print_log_path, 'a') as log_file:
            log_file.write('Computing best threshold for macro F1-score based on validation dataset:\n')
    for t in thresholds:
        pred_labels = (pred_score_matrix[:, 1] >= t).astype(int)
        report_dict = compute_classification_report(pred_labels, true_labels, labels=labels, verbose=False)
        macro_f1 = report_dict["macro avg"]["f1-score"]
        if print_log_path is not None:
            with open(print_log_path, 'a') as log_file:
                log_file.write(f'Threshold: {t:.2f}, Macro F1-score: {macro_f1:.4f}\n')
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = t
    return best_threshold, best_macro_f1

def compute_metrics(pred_labels, pred_score_matrix, true_labels, label='', prefix='', print_log_path=None):
    num_classes = pred_score_matrix.shape[1]
    # if true_labels is one-hot encoded or continuous, convert to class indices
    if true_labels.shape[0] != true_labels.size:
        true_labels = np.argmax(true_labels, axis=1)
    metrics_dict = {}
    try:
        # Handle binary and multiclass cases
        if pred_score_matrix.shape[1] > 2:  # Multiclass
            roc_val = roc_auc_score(true_labels, pred_score_matrix, average='macro', multi_class='ovr') # one vs. rest
        else:  # Binary
            roc_val = roc_auc_score(true_labels, pred_score_matrix[:, 1], average='macro') # roc-auc for positive class is same as negative class for binary
        metrics_dict[f'{prefix}{label}-roc-macro'] = roc_val  # Store as list
    except ValueError as e:
        print(f"Warning: ROC AUC computation failed: {e}")
        metrics_dict[f'{prefix}{label}-roc-macro'] = None  # Assign None as a list

    if label == 'test':
        report_dict = compute_classification_report(pred_labels, true_labels, labels=np.arange(num_classes), print_log_path=print_log_path)
        metrics_dict[f'{prefix}macro-avg F1-score'] = report_dict["macro avg"]["f1-score"]
        metrics_dict[f'{prefix}support'] = report_dict["macro avg"]["support"]
        # Add detailed metrics for each label
        for idx in np.arange(pred_score_matrix.shape[1]):
            metrics_dict[f'{prefix}Label={idx}-f1-score'] = report_dict[str(idx)]['f1-score']
            metrics_dict[f'{prefix}Label={idx}-support'] = report_dict[str(idx)]['support']
            metrics_dict[f'{prefix}Label={idx}-precision'] = report_dict[str(idx)]['precision']
            metrics_dict[f'{prefix}Label={idx}-recall'] = report_dict[str(idx)]['recall']
    if label == 'val':
        best_threshold, best_macro_f1 = compute_best_threshold_for_macro_f1(pred_score_matrix=pred_score_matrix, true_labels=true_labels,
                                                                            labels=np.arange(num_classes), print_log_path=print_log_path)
        metrics_dict['best-threshold'] = best_threshold
        metrics_dict['best-valid-macro-f1'] = best_macro_f1
    return metrics_dict

def compute_stats_2d(features):
    if features.numel() == 0 or features.size(0) <= 1:
        # Return zeros for mean, ones for std if not enough data
        dummy_shape = (1, features.size(-1)) if features.size(-1) > 0 else (1, 1)
        return torch.zeros(dummy_shape), torch.ones(dummy_shape)
    features_flat = features.reshape(-1, features.size(-1))
    mean = features_flat.mean(dim=0, keepdim=True)
    std = features_flat.std(dim=0, keepdim=True, unbiased=False) + 1e-6  # Use unbiased=False
    return mean, std

def compute_mean_std(data_set, is_multimodal=False):
    if is_multimodal:
        features1, features2, features3, _, _ = data_set.tensors
        mean1, std1 = compute_stats_2d(features1)
        mean2, std2 = compute_stats_2d(features2)
        mean3, std3 = compute_stats_2d(features3)
        return (mean1, mean2, mean3), (std1, std2, std3)

    features = data_set.tensors[0]
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-6
    return mean, std

def perform_zscore_normalization(data_set, mean, std, is_multimodal=False):
    if is_multimodal:
        features1, features2, features3, labels, group_labels = data_set.tensors
        if features1.shape[1] == 0:
            norm_features1 = features1
        else:
            assert features1.dim() == 3, "Expected features1 to be 3D"
            assert mean[0].dim() == 1 and std[0].dim() == 1, "Expected mean[0] and std[0] to be 1D"
            norm_features1 = (features1 - mean[0].unsqueeze(0).unsqueeze(0)) / std[0].unsqueeze(0).unsqueeze(0)
        
        if features2.shape[1] == 0:
            norm_features2 = features2
        else:
            assert features2.dim() == 3, "Expected features2 to be 3D"
            assert mean[1].dim() == 1 and std[1].dim() == 1, "Expected mean[1] and std[1] to be 1D"
            norm_features2 = (features2 - mean[1].unsqueeze(0).unsqueeze(0)) / std[1].unsqueeze(0).unsqueeze(0)
        
        if features3.shape[1] == 0:
            norm_features3 = features3
        else:
            assert features3.dim() == 2, "Expected features3 to be 2D"
            assert mean[2].dim() == 1 and std[2].dim() == 1, "Expected mean[2] and std[2] to be 1D"
            norm_features3 = (features3 - mean[2].unsqueeze(0)) / std[2].unsqueeze(0)
        # norm_features1 = norm_features1.squeeze(0)
        # norm_features2 = norm_features2.squeeze(0)
        # norm_features3 = norm_features3.squeeze(0)
        return TensorDataset(norm_features1, norm_features2, norm_features3, labels, group_labels)

    features, labels = data_set.tensors
    norm_features = (features - mean) / std
    return TensorDataset(norm_features, labels)

def perform_zscore_normalization_pretraining(data_set, mean, std):
    norm_features = (data_set - mean) / std
    return norm_features
