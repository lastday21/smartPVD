import pandas as pd
from config import PPD_BASELINE_DAYS, PPD_REL_THRESH, PPD_MIN_EVENT_DAYS


def detect_ppd_events(ppd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect PPD percentage-drop events in PPD data.

    Parameters
    ----------
    ppd_df : pd.DataFrame
        DataFrame with columns ['date', 'well', 'q_ppd'], where 'date' is datetime-like.

    Returns
    -------
    events_df : pd.DataFrame
        DataFrame of detected events with columns:
        ['well', 'start_date', 'end_date', 'duration_days', 'baseline', 'min_q', 'relative_drop']
    """
    events = []

    # Ensure date column is datetime and sorted
    ppd_df = ppd_df.copy()
    ppd_df['date'] = pd.to_datetime(ppd_df['date'])

    for well, group in ppd_df.groupby('well'):
        df = group.sort_values('date').set_index('date')

        # Calculate rolling baseline excluding current day
        baseline = df['q_ppd'] \
            .rolling(window=PPD_BASELINE_DAYS, min_periods=PPD_BASELINE_DAYS) \
            .mean().shift(1)
        df['baseline'] = baseline

        # Relative drop: fraction below baseline
        df['relative_drop'] = 1 - df['q_ppd'] / df['baseline']

        # Identify where the drop exceeds threshold
        df['is_event'] = df['relative_drop'] >= PPD_REL_THRESH

        # Label contiguous runs of events
        df['event_group'] = (df['is_event'] != df['is_event'].shift(1)).cumsum()

        # Iterate over each event run
        for grp, sub in df[df['is_event']].groupby('event_group'):
            start = sub.index.min()
            end = sub.index.max()
            duration = (end - start).days + 1

            # Check minimum duration
            if duration >= PPD_MIN_EVENT_DAYS :
                baseline_val = sub['baseline'].iloc[0]
                min_q = sub['q_ppd'].min()
                rel_drop_val = sub['relative_drop'].max()

                events.append({
                    'well': well,
                    'start_date': start,
                    'end_date': end,
                    'duration_days': duration,
                    'baseline': baseline_val,
                    'min_q': min_q,
                    'relative_drop': rel_drop_val
                })

    return pd.DataFrame(events)


if __name__ == '__main__':

    try:
        df = pd.read_csv('clean_data/ppd_clean.csv')
        events = detect_ppd_events(df)
        print(events.head())
    except Exception:
        pass