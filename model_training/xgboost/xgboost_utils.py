import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import json
random_seed = 100

# assumes two classes
def upsample_minority_class(features, labels):
    # oversample = SMOTE(sampling_strategy='minority', random_state=random_seed)
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=random_seed)
    features_over, labels_over = oversample.fit_resample(features, labels)
    print(f'After upsampling: {Counter(labels_over)}')
    return features_over, labels_over

# assumes two classes
def downsample_majority_class(features, labels):
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=random_seed)
    features_under, labels_under = undersample.fit_resample(features, labels)
    print(f'After downsampling: {Counter(labels_under)}')
    return features_under, labels_under

# """Load the data from NPZ files
# the data object has keys 'features', 'label_values' which are scores from 0-100,
# 'pid_labels' which are participant IDs, and 'day_labels' which are day of data collection ID
# """
# def load_data(npz_path, require_shuffle=True, cv_type='train', random_seed=100,
#                             include_prev_label=False, include_demo_info=False):
#     print(f'Note: including prev label: {include_prev_label}, including demographic: {include_demo_info}')
#     data = np.load(npz_path)
#     features = data['features']
#     labels = data['labels']
#     if cv_type == 'loso':
#         group_labels = data['pid_labels']
#     elif cv_type in ['lodo', 'time-series']:
#         group_labels = data['day_labels']
#     else:
#         group_labels = -1
#     features[np.isinf(features)] = np.nan
#     feature_start_idx = 0 if include_demo_info else 2
#     feature_end_idx = features.shape[1] if include_prev_label else features.shape[1]-1
#     print(f'Start idx: {feature_start_idx}, End idx: {feature_end_idx}')
#     features = features[:, feature_start_idx:feature_end_idx]
#     if require_shuffle:
#         # features, labels = shuffle(features, labels, random_state=random_seed)
#         features, labels, group_labels = shuffle(features, labels, group_labels, random_state=random_seed)
#     # features = np.hstack((features, labels.reshape(-1,1))) # debug: adding label as a feature
#     # print statistics
#     print(f'Data: {cv_type}, Feature Dim: {features.shape}, \
#             Labels Split: {Counter(labels)}')
#     return features, labels, group_labels

# def convert_prob_to_labels(pred_prob_matrix, threshold=-1):
#     if threshold != -1:
#         # it's a binary classification
#         labels =  (pred_prob_matrix[:,1] > threshold).astype(int)
#     else:
#         labels = np.argmax(pred_prob_matrix, axis=1)
#     return labels

# def compute_roc_auc_score(true_labels, pred_prob_matrix, num_class):
#     auc_score = {}
#     if num_class == 2:
#         auc_score['micro'] = roc_auc_score(true_labels, pred_prob_matrix[:,1])
#         auc_score['macro'] = -1
#     else:
#         lb = LabelBinarizer()
#         lb.fit(np.arange(num_class))
#         true_label_onehot = lb.transform(true_labels)
#         auc_score['micro'] = roc_auc_score(true_label_onehot, pred_prob_matrix, average='micro', multi_class='ovr')
#         auc_score['macro'] = roc_auc_score(true_label_onehot, pred_prob_matrix, average='macro', multi_class='ovr')
#     return auc_score

def compute_accuracy_dict(train_acc, valid_acc, test_acc):
    acc_dict = {
        'train-acc':  [train_acc],
        'valid-acc': [valid_acc]
    }
    # if report_on == 'test':
    acc_dict['test-acc'] = [test_acc]
    return acc_dict

def compute_roc_dict(train_roc, valid_roc, test_roc, num_class):
    if num_class == 2:
        roc_dict = {
            'train-roc-macro': [train_roc['macro']],
            'valid-roc-macro': [valid_roc['macro']]
        }
        # if report_on == 'test':
        roc_dict['test-roc-macro'] = [test_roc['macro']]
    else:
        roc_dict = {
            'train-roc-micro': [train_roc['micro']],
            'valid-roc-micro': [valid_roc['micro']],
            'train-roc-macro': [train_roc['macro']],
            'valid-roc-macro': [valid_roc['macro']]
        }
        # if report_on == 'test':
        roc_dict['test-roc-micro'] = [test_roc['micro']]
        roc_dict['test-roc-macro'] = [test_roc['macro']]
    return roc_dict

def compute_training_loss_dict(eval_results, label_list, loss_type, best_iteration, threshold=-1):
    assert len(label_list) == 2
    loss_dict = {
        'Initial Train Loss': [eval_results[label_list[0]][loss_type][0]],
        'Final Train Loss': [eval_results[label_list[0]][loss_type][best_iteration]],
        'Initial Valid Loss': [eval_results[label_list[1]][loss_type][0]],
        'Final Valid Loss': [eval_results[label_list[1]][loss_type][best_iteration]],
        'Number of Rounds': [best_iteration]
    }
    if threshold != -1:
        loss_dict['threshold'] = threshold
    return loss_dict

def compute_precision_recall_dict(test_metrics, num_class, chosen_label):
    pr_dict = {
        'macro-avg F1-score': [test_metrics['macro avg']['f1-score']],
        'support': [test_metrics['macro avg']['support']]
    }
    for idx in np.arange(num_class):
        pr_dict[f'Label={idx}-f1-score'] =  [test_metrics[f'{chosen_label}={idx}']['f1-score']]
        pr_dict[f'Label={idx}-support'] = [test_metrics[f'{chosen_label}={idx}']['support']]
    return pr_dict

# def pretty_print_dict(d):
#     def convert_numpy(obj):
#         """ Convert NumPy types to native Python types for JSON serialization """
#         if isinstance(obj, (np.int_, np.int32, np.int64)):
#             return int(obj)
#         elif isinstance(obj, (np.float_, np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
#             return obj.tolist()
#         raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

#     print(json.dumps(d, indent=4, sort_keys=True, default=convert_numpy))


