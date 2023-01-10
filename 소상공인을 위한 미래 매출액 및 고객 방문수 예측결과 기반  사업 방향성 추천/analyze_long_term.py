# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt
import numpy as np


# 1. 분석데이터에서 고객의 사업장에 관련된 데이터 추출하기

# 입력받은 사업장의 장소와 서비스
my_loc, my_ser = sys.argv[1:3]

# 기본경로 알아내기)
if len(os.path.split(sys.argv[0])[0]) > 0:
    root = os.path.split(sys.argv[0])[0]
else:
    root = os.getcwd()
# print(root)
# .\ml_files or D:\jupyter\캡스톤\ml_files

# 분석에 사용할 데이터
origin_df = pd.read_csv(
    f'{root}/data/서울시우리마을가게상권분석서비스_kor_2014_2021.csv', index_col=0)
df_col_eng = ["thsmon_selng_amt", "thsmon_selng_co", "mdwk_selng_amt", "wkend_selng_amt", "mon_selng_amt", "tues_selng_amt", "wed_selng_amt", "thur_selng_amt", "fri_selng_amt", "sat_selng_amt", "sun_selng_amt", "tmzon_00_06_selng_amt", "tmzon_06_11_selng_amt", "tmzon_11_14_selng_amt", "tmzon_14_17_selng_amt", "tmzon_17_21_selng_amt", "tmzon_21_24_selng_amt", "ml_selng_amt", "fml_selng_amt", "agrde_10_selng_amt", "agrde_20_selng_amt", "agrde_30_selng_amt", "agrde_40_selng_amt",
              "agrde_50_selng_amt", "agrde_60_above_selng_amt", "mdwk_selng_co", "wkend_selng_co", "mon_selng_co", "tues_selng_co", "wed_selng_co", "thur_selng_co", "fri_selng_co", "sat_selng_co", "sun_selng_co", "tmzon_00_06_selng_co", "tmzon_06_11_selng_co", "tmzon_11_14_selng_co", "tmzon_14_17_selng_co", "tmzon_17_21_selng_co", "tmzon_21_24_selng_co", "ml_selng_co", "fml_selng_co", "agrde_10_selng_co", "agrde_20_selng_co", "agrde_30_selng_co", "agrde_40_selng_co", "agrde_50_selng_co", "agrde_60_above_selng_co", "stor_co"]

# 위 데이터에서 해당 사업장의 장소와 서비스 항목만 추출
location = origin_df[origin_df.columns[5]].str.contains(my_loc)
service = origin_df[origin_df.columns[7]].str.contains(my_ser)
df = origin_df[location & service].copy()

# 데이터가 없을 경우 프로그램 종료
if df.shape[0] < 32:
    print('데이터가 없거나 부족합니다.')
    exit()


# 분석에 필요하지 않은 데이터 누락
df.drop([*df.columns[2:8], *df.columns[10:33]], axis=1, inplace=True)

# 연도 및 분기에 따라 데이터 합산
df[origin_df.columns[1]].replace(
    [1, 2, 3, 4], ['01-01', '04-01', '07-01', '10-01'], inplace=True)
df = df.groupby([origin_df.columns[0], origin_df.columns[1]]).sum()
df.reset_index(inplace=True)

# 연도와 분기를 하나의 날짜 데이터로 변환
df = df.astype({origin_df.columns[0]: 'str', origin_df.columns[1]: 'str'})
df['ds'] = df[origin_df.columns[0]].str.cat(df[origin_df.columns[1]], sep="-")
df.drop(df.columns[:2], axis=1, inplace=True)
df = df.reindex(columns=[df.columns[-1]]+[*df.columns[:-1]])

# 2. 향후 데이터 예측하기

# 결과데이터 저장경로 설정
save_dir = f'{root}/analyze_long_term_work/work_{sys.argv[1]}_{sys.argv[2]}'
if os.path.isfile(save_dir+'/completed.txt'):
    os.remove(save_dir+'/completed.txt')

# 저장경로 없으면 생성
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(save_dir+'/img'):
    os.mkdir(save_dir+'/img')
if not os.path.isdir(save_dir+'/forecast'):
    os.mkdir(save_dir+'/forecast')
if not os.path.isdir(save_dir+'/model'):
    os.mkdir(save_dir+'/model')
if not os.path.isdir(save_dir+'/accuracy'):
    os.mkdir(save_dir+'/accuracy')

result_df = pd.DataFrame()
acc_result_df = pd.DataFrame()
for i, v in enumerate(df.columns[1:], 1):
    # 모델 생성
    m = Prophet()
    m.daily_seasonality = False
    m.weekly_seasonality = False
    m.add_seasonality(name='quarter', period=365/4, fourier_order=10)
    m.yearly_seasonality = True

    # 적용시킬 데이터
    tmp_df = pd.DataFrame()
    tmp_df['ds'] = df['ds']
    tmp_df['y'] = df[df.columns[i]]

    # 값 예측
    m.fit(tmp_df)
    future = m.make_future_dataframe(periods=4, freq='QS')
    forecast = m.predict(future)
    result_df[v] = forecast['yhat']

    # 예측 결과 저장
    forecast.to_csv(save_dir+f'/forecast/{i}_{v}.csv')

    # 이미지 저장
    m.plot(forecast).savefig(save_dir+f'/img/{i}_{v}.png')
    m.plot_components(forecast).savefig(save_dir+f'/img/{i}_{v}_component.png')

    # 모델저장
    with open(save_dir+f'/model/{i}_{v}_model.json', 'w') as fout:
        fout.write(model_to_json(m))

    # cross validation 저장
    df_cv = cross_validation(m, initial='2190 days',
                             period='91.25 days', horizon='365 days')
    df_p = performance_metrics(df_cv)

    tmp_df = df_p.astype({'horizon': 'str'})
    tmp_df['horizon'] = tmp_df['horizon'].str[:-len(tmp_df['horizon'][0])+1]
    tmp_df = tmp_df.astype({'horizon': 'int'})
    cv_result_df = pd.DataFrame()
    for di in range(1, 5):
        cv_result_df = pd.concat([cv_result_df, df_p[tmp_df['mse'] == min(
            tmp_df[tmp_df['horizon'] == di*9]['mse'])]])
    cv_result_df.index = df['ds'][-4:]

    fig, axes = plt.subplots(2, 3, constrained_layout=True, figsize=(15, 10))
    for pi in range(2):
        for pj in range(3):
            if pi*3+pj+1 > len(cv_result_df.columns)-2:
                break
            col = cv_result_df.columns[pi*3+pj+1]
            pl_fig = cv_result_df[col].plot(
                kind='bar', ax=axes[pi, pj], title=f"{col}", grid=True, legend=None)
            pl_fig.get_yaxis().get_major_formatter().set_scientific(False)
    plt.savefig(save_dir+f'/img/{i}_{v}_cross_vali.png',
                dpi=300, bbox_inches='tight')
    if 'mape' in cv_result_df.columns:
        er = cv_result_df['mape']
    else:
        er = cv_result_df['mdape']
    acc_result_df = pd.concat(
        [acc_result_df, pd.DataFrame(abs(1-er)).mean()])


# 정확도
acc_result_df.iloc[1:].replace([np.inf, -np.inf], 0)
acc_result_df = pd.concat([acc_result_df.mean(), acc_result_df])
acc_result_df.columns = ['accuracy']
acc_result_df.index = ['ALL']+df_col_eng
acc_result_df[::-1].plot(kind='barh',
                         title='Accuracy by index', figsize=(3, 20))
plt.grid(True, axis='x')
plt.tick_params(top=True, labeltop=True)
plt.legend(loc=(1, 0.98))
plt.savefig(save_dir+f'/accuracy/acc.png', dpi=300, bbox_inches='tight')
acc_result_df.to_csv(save_dir+'/accuracy/acc.csv')

# 총 결과 저장
result_df.columns = df.columns[1:]
result_df['ds'] = forecast['ds']
result_df = result_df.astype({'ds': 'str'})
result_df['ds'] = result_df['ds'].str.replace('01-01', '1분기')
result_df['ds'] = result_df['ds'].str.replace('04-01', '2분기')
result_df['ds'] = result_df['ds'].str.replace('07-01', '3분기')
result_df['ds'] = result_df['ds'].str.replace('10-01', '4분기')
result_df = result_df.reindex(
    columns=[result_df.columns[-1]]+[*result_df.columns[:-1]])
result_df.to_csv(save_dir+'/result.csv')

print('분석에 성공하였습니다.')


# 완료를 나타내는 파일
with open(save_dir+'/completed.txt', 'w') as f:
    pass
