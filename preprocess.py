import numpy as np
from config import GAP_LIMIT, FREQ_THRESH

def preprocess_ppd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # шум: q=0 & есть p_cust → NaN
    mask = (df['q_ppd']==0) & (df['p_cust']>0)
    df.loc[mask,'q_ppd'] = np.nan
    return df

def preprocess_oil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # шум: q_oil=0 & freq>=FREQ_THRESH → NaN
    mask = (df['q_oil']==0) & (df['freq']>=FREQ_THRESH)
    df.loc[mask,'q_oil'] = np.nan
    return df

def resample_and_fill(series: pd.Series) -> pd.Series:
    # интерполяция до GAP_LIMIT дней; затем 0
    return series.resample('D').mean() \
                 .interpolate(limit=GAP_LIMIT, limit_direction='both') \
                 .fillna(0)
