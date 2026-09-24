import pandas as pd
df = pd.read_csv("../outputs/agri_s2_labels.csv")
print(df[["acq_date", "nbr_pre", "nbr_post", "post_scenes", "dnbr"]])