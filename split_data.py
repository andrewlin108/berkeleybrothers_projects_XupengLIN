import pandas as pd
import numpy as np
import os
if __name__ == '__main__':
    # underlying = 'CF'
    # underlying = 'OI'
    # underlying = 'RM'
    underlying = 'SR'
    file_name = 'opt_test_data/{}_options_data.parquet'.format(
        underlying)

    # 读取文件
    df = pd.read_parquet(file_name, engine='pyarrow')

    df = df.sort_values(by=['minute_str', 'strike']).reset_index(drop=True)
    df['t'] = df['maturity'] / 365
    df['t'] = df['t'].astype(float)

    if not os.path.exists('/Users/andrew/PycharmProjects/playground/boxiong_project/opt_test_data/{}'.format(underlying)):
        os.makedirs('/Users/andrew/PycharmProjects/playground/boxiong_project/opt_test_data/{}'.format(underlying))
    df['underlying_instr'] = df['Instrument'].apply(lambda x: x[:6])

    for temp_instr in df['underlying_instr'].unique():
        print(temp_instr)
        temp_df=df[df['underlying_instr'] == temp_instr]
        temp_df.to_parquet('/Users/andrew/PycharmProjects/playground/boxiong_project/opt_test_data/{}/{}_{}_options_data.parquet'.format(underlying,underlying,temp_instr),index=False)
        df_temp = pd.read_parquet('/Users/andrew/PycharmProjects/playground/boxiong_project/opt_test_data/{}/{}_{}_options_data.parquet'.format(underlying,underlying,temp_instr), engine='pyarrow')
        print()