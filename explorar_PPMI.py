#!/usr/bin/env python

import pandas as pd
import os

base_path = "/home/data/PPMI"
participants_file = os.path.join(base_path, "rawdata/participants.tsv")
curated_file = os.path.join(base_path, "documents/PPMI_Curated_Data_Cut_Public_20240729.xlsx")

# ----------------------- mirar tsv
print("Info on participants.tsv")
try:
    df_p = pd.read_csv(participants_file, sep='\t')
    print("Total subjects: ", len(df_p))

    # Look for a 'diagnosis' or 'group' column
    if 'diagnosis' in df_p.columns:
        print(df_p['diagnosis'].value_counts())
    else:
        print("Columns available:", df_p.columns.tolist())
except Exception as e:
    print("Could not read participants.tsv: ", e)

# ----------------------- mirar excel
print("Info on the excel")
try:
    df_c = pd.read_excel(curated_file)
    print(f"Total entries in curated data: {len(df_c)}")
    
    # Common PPMI diagnosis labels: PD (Parkinson's), HC (Healthy Control), SWEDD
    if 'COHORT_DEFINITION' in df_c.columns:
        print(df_c['COHORT_DEFINITION'].value_counts())
    else:
        # If the column name is different lemme see them
        print("Columns available:", df_c.columns.tolist()[:10], "...(truncated)")

except Exception as e:
    print(f"Could not read Excel file: {e}")
