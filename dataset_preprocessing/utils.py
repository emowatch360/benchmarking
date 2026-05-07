from datetime import datetime, timedelta

# A small constant to account for floating-point precision issues.
DELTA = 1e-8

def compute_day_str(chosen_day, first_date_str):
    """
    Given a chosen day identifier (e.g. 'd01') and a start date,
    compute the corresponding date string in the format YYYY-MM-DD.
    """
    first_date = datetime.strptime(first_date_str, "%Y/%m/%d")
    # Assumes chosen_day is like 'd01' where the last two digits indicate the day index (1-indexed)
    req_day_int = int(chosen_day[-2:]) - 1
    req_date = first_date + timedelta(days=req_day_int)
    return req_date.strftime("%Y-%m-%d")

def compute_day_id(date_str, first_date_str, fmt= "%Y-%m-%d"):
    """
    Given a date string (e.g., '2024-11-06') and the start date string (e.g., '2024-11-02'),
    compute the corresponding day identifier in the format 'dXX' (e.g., 'd05').
    """
    date = datetime.strptime(date_str, fmt)
    first_date = datetime.strptime(first_date_str, "%Y/%m/%d")
    delta_days = (date - first_date).days + 1  # +1 because 'd01' corresponds to first_date
    return f'd{delta_days:02d}'

def convert_to_minutes(timestamp, fmt='YYYY-MM-DDTHH:MM:SS.sss'):
    """
    Convert a timestamp string to the number of minutes since the start of the day.
    Supported formats:
      - 'YYYY-MM-DDTHH:MM:SS.sss'
      - 'YYYY/MM/DD HH:MM:SS'
      - 'HH:MM'
    Returns an integer count of minutes.
    """
    if fmt == 'YYYY-MM-DDTHH:MM:SS.sss':
        # Extract the time part 'HH:MM:SS'
        time_str = timestamp[11:19]
        hour, minute, _ = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'YYYY/MM/DD HH:MM:SS' or fmt == 'YYYY-MM-DD HH:MM:SS':
        time_str = timestamp.split(' ')[1]
        hour, minute, _ = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'YYYY/MM/DD HH:MM':
        time_str = timestamp.split(' ')[1]
        hour, minute = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
    elif fmt == 'HH:MM':
        hour, minute = map(int, timestamp.split(':'))
        total_minutes = hour * 60 + minute
    else:
        raise ValueError(f"Incorrect timestamp format: {fmt}")
    return total_minutes

def compute_day_and_time_indices(timestamp, first_date_str, fmt='YYYY/MM/DD HH:MM'):
    time_index = convert_to_minutes(timestamp, fmt)
    day_index = compute_day_id(timestamp.split(' ')[0], first_date_str)
    return day_index, time_index