import pandas as pd
from config import (
    OIL_CHECK_DAYS,
    OIL_DELTA_P_THRESH,
    OIL_EXTEND_DAYS,
    MAX_EVENT_DAYS
)


def analyze_oil_events(oil_df: pd.DataFrame, ppd_events: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze oil response for each PPD event.

    For each PPD event (with start and end dates), examines oil pressure and oil rate
    over an initial window of OIL_CHECK_DAYS. If the change in intake pressure (P_priem)
    exceeds OIL_DELTA_P_THRESH during those days, the window is extended to
    OIL_EXTEND_DAYS (but not beyond MAX_EVENT_DAYS or the PPD event duration).

    Parameters
    ----------
    oil_df : pd.DataFrame
        DataFrame with columns ['date', 'well', 'q_oil', 'p_priem'] and daily data.
    ppd_events : pd.DataFrame
        DataFrame with PPD events from detect_ppd_events(), columns:
        ['well', 'start_date', 'end_date', ...]

    Returns
    -------
    oil_events : pd.DataFrame
        DataFrame with one record per PPD event, columns:
        ['well', 'ppd_start', 'ppd_end', 'oil_end', 'duration_days', 'delta_q', 'delta_p']
    """
    records = []

    # Ensure date is datetime and set index for oil_df
    oil = oil_df.copy()
    oil['date'] = pd.to_datetime(oil['date'])
    oil = oil.sort_values('date').set_index('date')

    for ev in ppd_events.itertuples():
        well = ev.well
        start = pd.to_datetime(ev.start_date)
        ppd_end = pd.to_datetime(ev.end_date)

        # take data for this well
        series = oil[oil['well'] == well]

        # initial window
        initial_end = start + pd.Timedelta(days=OIL_CHECK_DAYS - 1)
        # Clip to series index
        if initial_end not in series.index:
            # find nearest available date <= initial_end
            available = series.loc[:initial_end]
            if available.empty:
                continue
            initial_end = available.index.max()

        # Compute pressure change in initial window
        p_start = series.at[start, 'p_priem'] if start in series.index else series.loc[start:].iloc[0]['p_priem']
        p_init_end = series.at[initial_end, 'p_priem']
        delta_p_init = abs(p_init_end - p_start)

        # decide extension
        if delta_p_init >= OIL_DELTA_P_THRESH:
            extend_end = start + pd.Timedelta(days=OIL_EXTEND_DAYS - 1)
        else:
            extend_end = initial_end

        # apply maximum limits
        max_limit = start + pd.Timedelta(days=MAX_EVENT_DAYS - 1)
        final_end = min(extend_end, max_limit, ppd_end)

        # ensure final_end exists
        if final_end not in series.index:
            available = series.loc[:final_end]
            if available.empty:
                continue
            final_end = available.index.max()

        # compute deltas
        q_start = series.at[start, 'q_oil'] if start in series.index else series.loc[start:].iloc[0]['q_oil']
        q_final = series.at[final_end, 'q_oil']
        delta_q = q_final - q_start

        p_final = series.at[final_end, 'p_priem']
        delta_p = p_final - p_start

        duration = (final_end - start).days + 1

        records.append({
            'well': well,
            'ppd_start': start,
            'ppd_end': ppd_end,
            'oil_end': final_end,
            'duration_days': duration,
            'delta_q': delta_q,
            'delta_p': delta_p
        })

    return pd.DataFrame(records)


if __name__ == '__main__':
    # Quick smoke test (requires clean_data/*.csv and events_ppd output)
    try:
        raw_oil = pd.read_csv('clean_data/oil_clean.csv')
        from events_PPD import detect_ppd_events

        raw_ppd = pd.read_csv('clean_data/ppd_clean.csv')
        ppd_events = detect_ppd_events(raw_ppd)
        oil_events = analyze_oil_events(raw_oil, ppd_events)
        print(oil_events.head())
    except Exception:
        pass