import numpy as np # type: ignore
import pandas as pd # type: ignore
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve # type: ignore
from sklearn.utils.class_weight import compute_sample_weight # type: ignore
from sklearn.preprocessing import LabelBinarizer # type: ignore

class XGBoostTrainer:
    def __init__(self, params, num_class):
        self.params = params
        self.num_class = num_class
        self.threshold = 0.5
        self.model = self.initialize_model()
    
    def initialize_model(self):
        model = xgb.XGBClassifier(
            objective=self.params['objective'], #binary:logistic for 2 classes
            eval_metric=self.params['eval_metric'], # 'auc', for two -- logloss, last metric used for early stopping
            booster=self.params['booster'],
            use_label_encoder=False,
            max_depth=self.params['max_depth'],
            min_child_weight=self.params['min_child_weight'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            gamma=self.params['gamma'],
            learning_rate=self.params['learning_rate'],
            n_estimators=self.params['n_estimators'],
            num_class=self.num_class,
            early_stopping_rounds=self.params['early_stopping'],
            reg_alpha=self.params['reg_alpha'],
            reg_lambda=self.params['reg_lambda'],
            verbosity=self.params['verbosity']
        )
        return model
    
    def get_eval_results(self):
        return self.model.evals_result()

    def get_best_iteration(self):
        return self.model.best_iteration

    def train(self, X_train, y_train, X_valid, y_valid):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        self.model.fit(X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_train, y_train), (X_valid, y_valid)], # early stopping always based on last dataset
                verbose=True)
    
    def evaluate(self, X_data):
        # predict_proba automatically utilizes the best iteration
        pred_score_matrix = self.model.predict_proba(X_data)
        return pred_score_matrix

    def compute_feature_importance(self, feature_names=None, importance_type="gain"):
        """
        Computes and optionally plots feature importance from an XGBoost model.
        Parameters:
        - xgb_model: Trained XGBoost model (`xgb.Booster`, `xgb.XGBClassifier`, or `xgb.XGBRegressor`).
        - feature_names: List of feature names (optional, defaults to 'f0', 'f1', ...).
        - importance_type: One of ['weight', 'gain', 'cover']. Default is 'gain'.
        Returns:
        - feature_importance_df: Pandas DataFrame containing feature importance scores.
        """
        # Get feature importance scores from XGBoost
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        # If feature names are not provided, default to XGBoost feature indices
        # if feature_names is None:
        #     feature_names = [f"f{i}" for i in range(0, len(importance_dict))]  # Create generic feature names
        # Ensure feature names match feature numbers
        # importance_dict = {feature_names[int(k[1:])] if k[1:].isdigit() else k: v for k, v in importance_dict.items()}
        # Convert to DataFrame
        feature_importance_df = pd.DataFrame(
            list(importance_dict.items()), columns=["Feature", "Importance"]
        ).sort_values(by="Importance", ascending=False)
        feature_importance_df = feature_importance_df.reset_index(drop=True)
        feature_importance_top10_dict = {'Feature Importance' :'-'.join(feature_importance_df['Feature'][:10])}
        return feature_importance_df, feature_importance_top10_dict
    
    def convert_scores_to_labels(self, pred_score_matrix):
        labels = np.argmax(pred_score_matrix, axis=1) # always go with argmax
        return labels

    def compute_roc_auc_score(self, true_labels, pred_score_matrix):
        auc_score = {}
        # if np.unique(true_labels).size < 2:
        #     print('!! Cannot compute ROC')
        #     auc_score['micro'] = np.nan
        #     auc_score['macro'] = np.nan
        if self.num_class == 2:
            auc_score['micro'] = roc_auc_score(true_labels, pred_score_matrix[:,1], average='micro')
            auc_score['macro'] = roc_auc_score(true_labels, pred_score_matrix[:,1], average='macro')
        else:
            lb = LabelBinarizer()
            lb.fit(np.arange(self.num_class))
            true_label_onehot = lb.transform(true_labels)
            auc_score['micro'] = roc_auc_score(true_label_onehot, pred_score_matrix, average='micro', multi_class='ovr')
            auc_score['macro'] = roc_auc_score(true_label_onehot, pred_score_matrix, average='macro', multi_class='ovr')
        return auc_score

    def compute_metrics(self, true_labels, pred_labels, pred_score_matrix, chosen_label, data_type,
                        print_log_path=None):
        accuracy_val = accuracy_score(true_labels, pred_labels)
        roc_val = self.compute_roc_auc_score(true_labels, pred_score_matrix)
        if np.unique(true_labels).size < self.num_class:
            print(f'!! Not enough classes in {data_type} data for {chosen_label}')
        overall_metrics = classification_report(true_labels, pred_labels, labels=np.arange(self.num_class),
                        target_names=[f'{chosen_label}={idx}' for idx in range(self.num_class)], output_dict=True)
        if data_type == 'test' and print_log_path is not None:
            with open(print_log_path, 'a') as log_file:
                log_file.write(f'Classification Report for {chosen_label} on {data_type} data:\n')
                log_file.write(classification_report(
                    true_labels, pred_labels, labels=np.arange(self.num_class),
                    target_names=[f'{chosen_label}={idx}' for idx in range(self.num_class)], output_dict=False
                ))
                log_file.write('\n\n')
        return accuracy_val, roc_val, overall_metrics
    






