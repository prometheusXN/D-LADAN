import random
import json

small_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/data/train_law_charge_case.json'
large_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/big_data/train_law_charge_case.json'

law_charge_case_small = json.load(open(small_path, 'r'))
law_charge_case_large = json.load(open(large_path, 'r'))

print(len(law_charge_case_small.keys()))
print(len(law_charge_case_large.keys()))