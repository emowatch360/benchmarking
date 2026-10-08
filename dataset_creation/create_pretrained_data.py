import os
from data_config import *
from emowatch_utils import *

# labels = ['valence_current', 'arousal_current', 'stressed_current'] #['stress_level'] #['valence_current', 'arousal_current', 'stressed_current'] #, 'tired_current'] #, # stress_level
ds_info = get_dataset_dict('emowatch')
dataset_country = 'jp'
assert ds_info['dataset-country'].lower() == dataset_country.lower(), "Dataset country mismatch!"
ds_info['feature-dim'] = '2D' # always 2D
num_class = 2
epoch_list = [30] #, 10]

for epoch in epoch_list:
    ds_info['duration'] = epoch
    ds_info['save-folder'] = os.path.join(ds_info['data-folder'], 'numpy_arrays_pretraining', f'{num_class}class', f'{dataset_country}_features28_epoch{ds_info['duration']}')
    load_emowatch_pretraining(ds_info=ds_info, save_data=True)
    # sleep data
    load_emowatch_pretraining_sleep(ds_info=ds_info, save_data=True)