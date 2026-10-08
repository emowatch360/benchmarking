import os
import pandas as pd

data_folder = '/data/emowatch/emowatch/'
dates_path = os.path.join(data_folder, 'jp_start_end_dates.csv')
detailed_dates_path = os.path.join(data_folder, 'detailed_jp_start_end_dates.csv')
dates_df = pd.read_csv(dates_path)

# current dates_df columns: ['pid', 'start date', 'end date']
# new detailed_dates_df should have 'pid', 'd01', ... 'd28', where each day column has the corresponding date string
# start date is d01, and end date is d28

detailed_dates_data = {'pid': []}
for day_idx in range(1, 29):
    day_col = f'd{day_idx:02d}'
    detailed_dates_data[day_col] = []
for idx, row in dates_df.iterrows():
    pid = row['pid']
    # only keep pids that start with MM
    if not str(pid).startswith('MM'):
        continue
    start_date_str = row['start date']  # 'YYYY-MM-DD'
    end_date_str = row['end date']      # 'YYYY-MM-DD'
    start_date = pd.to_datetime(start_date_str, format='%Y/%m/%d')
    end_date = pd.to_datetime(end_date_str, format='%Y/%m/%d')
    detailed_dates_data['pid'].append(pid)
    for day_idx in range(1, 29):
        day_col = f'd{day_idx:02d}'
        current_date = start_date + pd.Timedelta(days=day_idx - 1)
        if current_date > end_date:
            raise ValueError(f'For pid={pid}, computed date for {day_col} exceeds end date.')
        else:
            detailed_dates_data[day_col].append(current_date.strftime('%Y/%m/%d'))
        # check that end date matches for d28
        if day_idx == 28:
            assert current_date.strftime('%Y/%m/%d') == end_date_str, f'End date mismatch for pid={pid}'
detailed_dates_df = pd.DataFrame(detailed_dates_data)
detailed_dates_df.to_csv(detailed_dates_path, index=False)
print(f'Detailed JP start-end dates saved to {detailed_dates_path}')