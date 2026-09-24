import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
 
EARTH_RADIUS_M = 6371000
EVENT_RADIUS_M = 500       
MAX_TIME_GAP_DAYS = 60     
 
 
def compute_event_ids(df, radius_m=EVENT_RADIUS_M, max_time_gap_days=MAX_TIME_GAP_DAYS):
    
    coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
    eps_rad = radius_m / EARTH_RADIUS_M
 
    db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine", algorithm="ball_tree")
    spatial_cluster_ids = db.fit_predict(coords_rad)
 
    dates = pd.to_datetime(df["acq_date"]).to_numpy()
    event_ids = np.full(len(df), -1, dtype=int)
    next_event_id = 0
 
    for spatial_id in np.unique(spatial_cluster_ids):
        idxs = np.where(spatial_cluster_ids == spatial_id)[0]
 
        
        order = np.argsort(dates[idxs])
        sorted_idxs = idxs[order]
        sorted_dates = dates[sorted_idxs]
 
        event_ids[sorted_idxs[0]] = next_event_id
        for i in range(1, len(sorted_idxs)):
            gap_days = (sorted_dates[i] - sorted_dates[i - 1]) / np.timedelta64(1, "D")
            if gap_days > max_time_gap_days:
                next_event_id += 1  
            event_ids[sorted_idxs[i]] = next_event_id
 
        next_event_id += 1  
    return event_ids
 