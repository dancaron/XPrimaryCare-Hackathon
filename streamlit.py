import sqlite3
import streamlit as st
import numpy as np
import pickle
import sklearn
import pandas as pd
import snowflake.connector
import gzip
import json

# Snowflake connection parameters
account = 'ckb61329.prod3.us-west-2.aws'
user = 'trentbuckholz'
password = 'Roshi321!'
warehouse = 'compute_wh'
database = 'TUVA_PROJECT_DEMO'
schema = 'TUVA_SYNTHETIC'

conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema=schema
)

cur = conn.cursor()
query = f"select distinct diagnosis_code_1 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_2 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_3 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_4 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_5 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_6 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_7 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_8 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_9 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_10 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_11 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_12 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_13 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_14 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_15 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_16 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_17 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_18 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_19 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_20 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_21 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_22 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_23 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_24 from tuva_synthetic.medical_claim UNION ALL select distinct diagnosis_code_25 from tuva_synthetic.medical_claim"
cur.execute(query)

# Retrieve the data
diag_codes_sql = cur.fetchall()

#Convert diagnosis codes into array
diag_codes = pd.DataFrame(diag_codes_sql).to_numpy().flatten()
diag_codes = list(diag_codes)
diag_codes = json.dumps(diag_codes[1:])

# Close the cursor and connection
cur.close()
conn.close()

cur.execute(query)
diag_codes_sql = cur.fetchall()
diag_codes = pd.DataFrame(diag_codes_sql).to_numpy().flatten()
diag_codes = list(diag_codes)
diag_codes = json.dumps(diag_codes[1:])
cur.close()
conn.close()

COLUMNS = ['DIAGNOSIS_CODE_1', 'DIAGNOSIS_CODE_2', ...]  # as it was in your original script

with gzip.open('model_pickle.gz', 'rb') as f:
    model = pickle.load(f)

with gzip.open('encoder_pickle.gz', 'rb') as f:
    encoder = pickle.load(f)

def return_med_prediction(diagnosis_list):
    if len(diagnosis_list) > 25:
        raise ValueError('only 25 conditions are allowed')
    diagnosis_list_final = []
    for i in range(25):
        if i < len(diagnosis_list):
            # Diagnosis needs to be encoded as int since model can only interpret numbers.
            diagnosis_list_final.append(encoder[diagnosis_list[i]])
        else:
            diagnosis_list_final.append(encoder[None])
    diagnosis_list_final.append(1)
    return encoder.inverse[model.predict(pd.DataFrame([diagnosis_list_final], columns=COLUMNS))[0]]


st.title('Diagnosis Predictor')
num_codes = st.number_input('Enter number of codes:', min_value=1, max_value=25, value=1, step=1)
submitted_diag_codes = []
for i in range(num_codes):
    code = st.text_input(f'Enter diagnosis code {i+1}:')
    submitted_diag_codes.append(code)
if st.button('Predict'):
    pred = return_med_prediction(submitted_diag_codes)
    st.write(f'Prediction: {pred}')
