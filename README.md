MGP-HATCN: A Hierarchcial attention based TCN Architecture for predicting sepsis in advance
==============================

Forked 
------------
This repo has been forked from https://github.com/mmr12/MGP-AttTCN. Most of the code is reused except the model architecture files in src/models/

Data 
------------
The dataset used is the MIMIC III dataset, fount at https://mimic.physionet.org.

Use
------------

STEP I: install dependencies 
`pip install -r requirements.txt`

STEP II: data extraction & preprocessing 
`python scr/data_processing/main.py [-h] -u SQLUSER -pw SQLPASS -ht HOST -db DBNAME -r SCHEMA_READ_NAME [-w SCHEMA_WRITE_NAME]`

STEP III: run the model

Credits 
------------
Credits to Margherita Rosnati & Vincent Fortuin for their implementation
Credits to M. Moor for sharing his code from https://arxiv.org/abs/1902.01659
