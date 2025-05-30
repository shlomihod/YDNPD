import os
import json
import itertools
import numpy as np
import pandas as pd
from ydnpd.harness.synthesis.src.utils import Dataset, Domain
from ydnpd.harness.synthesis.src.utils.general import get_min_dtype

def get_data(name, domain_name=None, root_path='./datasets/', cols='default'):
    df_path = os.path.join(root_path, "{}.csv".format(name))
    df = pd.read_csv(df_path)

    if domain_name is None:
        domain_name = "domain/{}-domain.json".format(name)
    domain_path = os.path.join(root_path, domain_name)
    config = json.load(open(domain_path))
    domain = Domain(config.keys(), config.values())

    # for saving memory
    dtype = get_min_dtype(sum(domain.config.values()))
    df = df.astype(dtype)

    data = Dataset(df, domain)

    if cols == 'default':
        cols = get_default_cols(name)
    if cols is None:
        return data
    data = data.project(cols)

    return data

def get_all_workloads(data, marginal):
    workloads = [p for p in itertools.combinations(data.domain.attrs, marginal)]
    return workloads

def get_rand_workloads(data, num_workloads, marginal, seed=0, check_size=False):
    prng = np.random.RandomState(seed)
    workloads = get_all_workloads(data, marginal)
    if check_size:
        total = data.df.shape[0]
        dom = data.domain
        workloads = [p for p in workloads if dom.size(p) <= total]
    if len(workloads) > num_workloads:
        workloads = [workloads[i] for i in prng.choice(len(workloads), num_workloads, replace=False)]
    return workloads

def get_default_cols(dataset):
    cols = None
    if dataset == 'adult_orig':
        cols = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    elif dataset == 'loans':
        cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment', 'annual_inc', 'dti',
                'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
                'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
                'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
                'delinq_amnt', 'pub_rec_bankruptcies', 'settlement_amount', 'settlement_percentage', 'settlement_term',
                'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'issue_d',
                'loan_status', 'purpose', 'zip_code', 'addr_state', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
                'last_credit_pull_d', 'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
                'settlement_date']
    elif dataset.startswith('adult'):
        if dataset.endswith('-reduced'):
            cols = ['sex', 'income>50K', 'race', 'marital-status',
                    'occupation', 'education',
                    'age'
                    ]
        else:
            cols = ['sex', 'income>50K', 'race', 'relationship', 'marital-status', 'workclass',
                    'occupation', 'education', 'native-country',
                    'capital-gain', 'capital-loss', 'hours-per-week',
                    'age'
                    ]
    elif dataset.startswith('acs'):
        if dataset.endswith('-reduced'):
            cols = ['SEX', 'FOODSTMP',
                    'RACWHT', 'RACASIAN', 'RACBLK', 'RACAMIND', 'RACPACIS', 'RACOTHER',
                    'DIFFEYE', 'DIFFHEAR', # 'DIFFPHYS', 'DIFFSENS',
                    'HCOVPRIV', 'HINSCAID', 'HINSCARE', # 'HCOVANY',
                    'OWNERSHP', # 'VETSTAT', 'CLASSWKR', 'ACREHOUS'
                    'EMPSTAT', # 'SCHLTYPE',
                    ]
        else:
            cols = ['VETWWII', 'AVAILBLE', 'MIGRATE1', 'MARRNO', 'GRADEATT', 'RACE', 'MARRINYR', 'EDUC', 'DIFFREM',
                    'VET75X90', 'EMPSTAT', 'VET47X50', 'MORTGAGE', 'VETVIETN', 'DIFFSENS', 'HCOVANY', 'LABFORCE',
                    'FOODSTMP', 'NCHILD', 'NSIBS', 'VETKOREA', 'VET90X01', 'RACWHT', 'RELATE', 'SEX', #'ROOMS',
                    'NMOTHERS', 'SCHLTYPE', 'DIFFEYE', 'VET55X64', 'SCHOOL', 'WIDINYR', 'MARST', 'VET01LTR', #'FAMSIZE',
                    'VEHICLES', 'WORKEDYR', 'VETDISAB', 'METRO', 'DIFFMOB', 'ACREHOUS', 'NFATHERS', #'LANGUAGE',
                    'NCHLT5', 'SPEAKENG', 'CLASSWKR', 'CITIZEN', 'VACANCY', 'RACASIAN', 'DIFFCARE', #'SEI', 'DEGFIELD',
                    'AGE', 'LOOKING', 'RACBLK', 'RACAMIND', 'DIFFPHYS', 'HINSCARE', # 'OCCSCORE', 'BUILTYR2', 'BEDROOMS',
                    'VETSTAT', 'MIGTYPE1', 'NCOUPLES', 'HISPAN', 'MULTGEN', 'DIFFHEAR', 'RACOTHER', 'HINSCAID', 'HINSVA',
                    'OWNERSHP', 'FERTYR', 'HCOVPRIV', 'DIVINYR', 'RACPACIS' # 'ELDCH', 'YNGCH', 'NFAMS',
                    ]
    return cols