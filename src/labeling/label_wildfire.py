import numpy as np
 
FRP_WILDFIRE_THRESHOLD = 15.0
FRP_STD_UNSTABLE_THRESHOLD = 5.0
NDVI_BURN_THRESHOLD = 0.2   
NBR_BURN_THRESHOLD = 0.1    
EVIDENCE_THRESHOLD = 1
 
 
def label_wildfire(df):
    
    is_high_frp = df["frp"] > FRP_WILDFIRE_THRESHOLD
    is_unstable = df["frp_std"] > FRP_STD_UNSTABLE_THRESHOLD
    frp_anomaly = (is_high_frp | is_unstable).astype(int)
 
    if "ndvi" in df.columns and "nbr" in df.columns:
        
        satellite_burn_evidence = (
            (df["ndvi"] < NDVI_BURN_THRESHOLD) | (df["nbr"] < NBR_BURN_THRESHOLD)
        ).astype(int)
    else:
        satellite_burn_evidence = np.zeros(len(df), dtype=int)
 
    evidence = frp_anomaly + satellite_burn_evidence
    return evidence >= EVIDENCE_THRESHOLD
