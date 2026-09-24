MINING_DISTNACE_THRESHOLD = 500
def label_mining(df):
    return df["mining_distance_m"]<=MINING_DISTNACE_THRESHOLD