# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from regex import W
import re

# 입력값 처리
# 입력받은 사업장의 장소와 서비스
my_loc, my_ser = sys.argv[1:3]
# 도표기준 최근 데이터 개수
if len(sys.argv) > 3:
    last_quaters = int(sys.argv[3])
else:
    last_quaters = 12

# 경로 처리
# 기본경로 알아내기)
if len(os.path.split(sys.argv[0])[0]) > 0:
    root = os.path.split(sys.argv[0])[0]
else:
    root = os.getcwd()

# print(root)
# .\ml_files or D:\jupyter\캡스톤\ml_files

# 작업 사전 준비
# 작업 완료여부 판별, 만약 안되면 프로그램 종료
work_argv = f'{sys.argv[1]}_{sys.argv[2]}'
if os.path.isfile(f'{root}/analyze_long_term_work/work_{work_argv}/completed.txt'):
    load_dir = f'{root}/analyze_long_term_work/work_{work_argv}/'
else:
    print('not yet')
    exit()

# 작업할 데이터 불러오기
df = pd.read_csv(load_dir+'result.csv', index_col=0)
df['ds'] = df['ds'].str[:-2]

search_dir = f"{root}/analyze_long_term_work/work_{work_argv}/forecast/"
forecasts = os.listdir(search_dir)
num_shops_df = df.iloc[-last_quaters:]['점포수']


# 1. 판매관련 업무량 분석
# 1-1. 요일별
# 분석할 항목
days = []  # ['월', '화', '수', '목', '금', '토', '일']
days_eng = []
look = []  # ['월요일_매출_건수', '화요일_매출_건수', '수요일_매출_건수', '목요일_매출_건수', '금요일_매출_건수', '토요일_매출_건수', '일요일_매출_건수'])
if df.iloc[-last_quaters:][df.columns[26]].sum():
    days += ['월', '화', '수', '목', '금']
    days_eng = ['Mon', 'Tue', 'Wes', 'Thu', 'Fri']
    look = [*df.columns[28:33]]
if df.iloc[-last_quaters:][df.columns[27]].sum():
    days += ['토', '일']
    days_eng += ['Sat', 'Sun']
    look += [*df.columns[33:35]]

# explain 1 : 요일별 수치 설명
my_df = df.iloc[-4:][df.columns[28:35]]
for i in range(7):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
my_df.index = df.iloc[-4:]['ds']
# 설명하기
say = '2022년 각 분기당 요일별 매출 건수는 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기의 경우 '
    for i, day in enumerate(days):
        say += f'{day}요일에 {int(my_df[look[i]][qu])}건, '
    say = say[:-2]
    say += '입니다.\n'
print(say)

# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain.txt', 'w') as f:
    f.write(say)

# 이미지 저장
my_df = df.iloc[-4:][df.columns[28:35]]
for i in range(7):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
fig = pd.DataFrame(my_df)
fig.columns = days_eng
fig.index = df.iloc[-4:]['ds']
fig.plot(kind='bar')
plt.title("Sales by day")
plt.legend(loc=(1, 0.5))
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain.png",
            dpi=300, bbox_inches='tight')


# explain 2 : 요일별 추세 설명
# LIS & LDS방식 이용하여 해당 트랜드 구하기
backup_a = []
trend = []
for index, elm in enumerate(look):
    for forecast in forecasts:
        if elm in forecast:
            break
    forecast_df = pd.read_csv(search_dir+forecast, index_col=0)

    a = list(forecast_df.iloc[-last_quaters:]['trend']/num_shops_df)
    backup_a.append(a)

    # LIS
    dp_i = [0 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[i] > a[j] and dp_i[i] < dp_i[j]:
                dp_i[i] = dp_i[j]
        dp_i[i] += 1

    # LDS
    dp_d = [1 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[j] > a[i]:
                dp_d[i] = max(dp_d[i], dp_d[j]+1)

    # 추세분석
    if max(dp_d) >= (last_quaters*2)//3 and max(dp_i) < (last_quaters*2)//3:
        trend.append([days[index], '-'])
    elif max(dp_d) < (last_quaters*2)//3 and max(dp_i) >= (last_quaters*2)//3:
        trend.append([days[index], '+'])
    else:
        trend.append([days[index], '='])

# 설명하기
say = '해당 가게/점포의 업무량이 증가하는지, 감소하는지 요일에 따라 분석하였습니다.\n'
last_say = ''
inc = []
dec = []
same = []
for i in trend:
    if i[1] == '+':
        inc.append(i[0])
    elif i[1] == '-':
        dec.append(i[0])
    else:
        same.append(i[0])

if inc:
    say += f'향후 업무량이 증가할 것으로 예상되는 요일은 { ", ".join(inc)} 입니다.\n'
    last_say += f'{", ".join(inc)}에는 더 많은 직원을, '
if dec:
    say += f'향후 업무량이 감소할 것으로 예상되는 요일은 {", ".join(dec)} 입니다.\n'
    last_say += f'{", ".join(dec)}에는 더 적은 직원을 '
if same:
    say += f'향후 업무량이 같을 것으로 예상되는 요일은 {", ".join(same)} 입니다.\n'
say += '만약 직원을 배치하신다면 ' + last_say + '배치하실 것을 추천합니다'
print(say)

# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain2.txt', 'w') as f:
    f.write(say)

# 도표 제작
fig = pd.DataFrame(backup_a).T
fig.columns = days_eng
fig.index = df.iloc[-last_quaters:]['ds']
fig.plot()
plt.title("days by sale")
plt.legend(loc=(1, 0.5))
plt.xlabel('quarter')
plt.ylabel('trend')
plt.grid(True)
plt.savefig(
    f"{root}/explain_long_term_work/{work_argv}/explain2.png", dpi=300, bbox_inches='tight')


# 1-2. 시간별
# 분석할 항목
# ['시간대_건수~06_매출_건수', '시간대_건수~11_매출_건수', '시간대_건수~14_매출_건수', '시간대_건수~17_매출_건수', '시간대_건수~21_매출_건수', '시간대_건수~24_매출_건수']
look = [*df.columns[35:41]]
times_eng = ['Dawn(00~06)', 'Morning(06~11)', 'Lunch(11~14)',
             'Afternoon(14~17)', 'Evening(17~21)', 'Nignt(21~24)']
times = [['00~06', '새벽'], ['06~11', '오전'], ['11~14', '점심'],
         ['14~17', '오후'], ['17~21', '저녁'], ['21~24', '밤  ']]

# explain 3 : 시간별 수치 설명
my_df = df.iloc[-4:][df.columns[35:41]]
for i in range(41-35):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
my_df.index = df.iloc[-4:]['ds']
# 설명하기
say = '2022년 각 분기당 시간별 매출 건수는 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기의 경우 '
    for i, time in enumerate(times):
        say += f'{time[1].rstrip()}에 {int(my_df[look[i]][qu])}건, '
    say = say[:-2]
    say += '입니다.\n'
print(say)
# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain3.txt', 'w') as f:
    f.write(say)
# 이미지 저장
my_df = df.iloc[-4:][df.columns[35:41]]
for i in range(41-35):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
fig = pd.DataFrame(my_df)
fig.columns = times_eng
fig.index = df.iloc[-4:]['ds']
fig.plot(kind='bar')
plt.title("Sales by time")
plt.legend(loc=(1, 0.5))
plt.xlabel('Quarter')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain3.png",
            dpi=300, bbox_inches='tight')

# explain 4 : 시간대별 추세
# LIS & LDS 이용하여 해당 트렌드 구하기
backup_a = []
trend = []

for index, elm in enumerate(look):
    for forecast in forecasts:
        if elm in forecast:
            break
    forecast_df = pd.read_csv(search_dir+forecast, index_col=0)

    a = list(forecast_df.iloc[-12:]['trend']/num_shops_df)
    backup_a.append(a)
#     print(elm, forecast, sum(a))
    # LIS
    dp_i = [0 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[i] > a[j] and dp_i[i] < dp_i[j]:
                dp_i[i] = dp_i[j]
        dp_i[i] += 1

    # LDS
    dp_d = [1 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[j] > a[i]:
                dp_d[i] = max(dp_d[i], dp_d[j]+1)
    # 추세분석
    if max(dp_d) >= (last_quaters*2)//3 and max(dp_i) < (last_quaters*2)//3:
        trend.append([times[index][1], '-'])
    elif max(dp_d) < (last_quaters*2)//3 and max(dp_i) >= (last_quaters*2)//3:
        trend.append([times[index][1], '+'])
    else:
        trend.append([times[index][1], '='])

# 설명하기
say = '해당 가게/점포의 업무량이 증가하는지, 감소하는지 시간대에 따라 분석하였습니다.\n'
for i, v in enumerate(trend):
    if v[1] == '+':
        state = '증가할'
    if v[1] == '-':
        state = '감소할'
    if v[1] == '=':
        state = '유지될'
    say += f'{times[i][1]}({times[i][0]}) : {state} 것으로 예상됩니다\n'

last_say = ''
inc = []
dec = []
same = []
for i in trend:
    if i[1] == '+':
        inc.append(i[0])
    elif i[1] == '-':
        dec.append(i[0])
    else:
        same.append(i[0])

if inc:
    last_say += f'{", ".join(inc).rstrip()}에는 더 많은 직원을, '
if dec:
    last_say += f'{", ".join(dec).rstrip()}에는 더 적은 직원을 '
say += '만약 직원을 배치하신다면 ' + last_say + '배치하실 것을 추천합니다'
print(say)

# 파일저장
with open(f'{root}/explain_long_term_work/{work_argv}/explain4.txt', 'w') as f:
    f.write(say)
# 도표 제작
fig = pd.DataFrame(backup_a).T
fig.columns = times_eng
fig.index = df.iloc[-last_quaters:]['ds']
fig.plot()
plt.title("Sales by time")
plt.legend(loc=(1, 0.5))
plt.xlabel('quarter')
plt.ylabel('trend')
plt.grid(True)
plt.savefig(
    f"{root}/explain_long_term_work/{work_argv}/explain4.png", dpi=300, bbox_inches='tight')

# 2. 상품관련 업무량 분석
# 2-1. 남녀별

# explain 5 : 남녀별 수치
# 분석할 항목
items = ['남성의 매출금액', '여성의 매출금액', '남성의 건수별매출액', '여성의 건수별매출액']
look = [*df.columns[18:20], *df.columns[41:43]]
my_df = df.iloc[-4:][[*df.columns[18:20], *df.columns[41:43]]]
for i in range(4):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
my_df.index = df.iloc[-4:]['ds']
# 설명하기
say = '2022년 각 분기당 남녀별 매출 금액은 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기의 경우 '
    for i, item in enumerate(items[:2]):
        num = format(int(my_df[look[i]][qu]), ',')
        say += f'{item}은 {num}원, '
    say = say[:-2]
    say += '입니다.\n'

say += '2022년 각 분기당 남녀별 매출건수는 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기의 경우 '
    for i, item in enumerate(items[2:], 2):
        num = format(int(my_df[look[i]][qu]), ',')
        say += f'{item}은 {num}건, '
    say = say[:-2]
    say += '입니다.\n'
print(say)
# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain5.txt', 'w') as f:
    f.write(say)
# 도표 제작
fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
fig1 = my_df[my_df.columns[:2]]
fig1.columns = ['Man', 'Woman']
fig1.index = df.iloc[-4:]['ds']


pl_fig = fig1.plot(
    kind='bar', ax=axes[0], title="Profit by sex", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = my_df[my_df.columns[2:]]
fig2.columns = ['Man', 'Woman']
fig2.index = df.iloc[-4:]['ds']

fig2.plot(kind='bar', ax=axes[1], title="Sales by sex", grid=True)
plt.legend(loc=(1, 0.5))
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain5.png",
            dpi=300, bbox_inches='tight')

# explain 6 : 남녀별 트렌드
# 트렌드 구하기
# LIS & LDS
backup_a = []
trend = []
for index, elm in enumerate(look):
    if index < 2:
        for forecast in forecasts:
            if elm in forecast:
                break
        forecast_df = pd.read_csv(search_dir+forecast, index_col=0)

        a = list(forecast_df.iloc[-12:]['trend']/num_shops_df)
    else:
        a = list((df.iloc[-16:][look[index-2]] /
                  df.iloc[-16:][elm]).diff(periods=1)[-12:])

    backup_a.append(a)
    dp_i = [0 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[i] > a[j] and dp_i[i] < dp_i[j]:
                dp_i[i] = dp_i[j]
        dp_i[i] += 1

    dp_d = [1 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[j] > a[i]:
                dp_d[i] = max(dp_d[i], dp_d[j]+1)

    if max(dp_d) >= 8 and max(dp_i) < 8:
        trend.append('-')
    elif max(dp_d) < 8 and max(dp_i) >= 8:
        trend.append('+')
    else:
        trend.append('=')


# 설명하기
say = '매출금액과 건수대비 판매액을 남성과 여성에 따라 분석하였습니다.\n'
state = []
for i, v in enumerate(trend):
    if v == '+':
        state.append('증가할')
    if v == '-':
        state.append('감소할')
    if v == '=':
        state.append('유지될')

for i, sex in enumerate(['남자', '여자']):
    say += f'{items[i]}은 {state[i]} 것이며, {items[i+2]}은 {state[i+2]} 것으로 예상됩니다\n즉 {sex}에게의 인기가 현재 '
    if trend[i] == '+':
        say += '성장중에 있으며 , '
    elif trend[i] == '-':
        say += '침체중에 있으며 '
    elif trend[i] == '=':
        say += '현상유지상태이며 '
    if trend[i+2] == '+':
        say += '특정 상품의 가치가 높아지고 있음을 뜻합니다. 해당 상품을 고급화 혹은 브랜드화 하는 것을 추천합니다.\n'
    elif trend[i+2] == '-':
        say += '전체 상품의 평균 가치가 떨어지고 있음을 뜻합니다. 미끼상품같이 이윤이 적은 상품이 많이 팔리는 것으로 보이므로 해당 상품의 판매를 중지하거나 줄이는 것을 추천합니다.\n'
    elif trend[i+2] == '=':
        say += '매우 평범한 상태를 의미합니다. 지금은 다양한 판매전략을 세우고 시행하기 가장 좋은 시기이므로 많은 판매전략을 시도해보시는 것이 좋을 것으로 생각됩니다.\n'
#


print(say)

# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain6.txt', 'w') as f:
    f.write(say)
# 도표 제작
fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
fig1 = pd.DataFrame(backup_a[:2]).T
fig1.columns = ['Man', 'Woman']
fig1.index = df.iloc[-12:]['ds']


pl_fig = fig1.plot(
    ax=axes[0], title="Profit trend by sex", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = pd.DataFrame(backup_a[2:]).T
fig2.columns = ['Man', 'Woman']
fig2.index = df.iloc[-12:]['ds']

fig2.plot(ax=axes[1], title="Sales trend by sex", grid=True)
plt.legend(loc=(1, 0.5))
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain6.png",
            dpi=300, bbox_inches='tight')

# 2-2 연령대별
# 분석할 항목
look = [*df.columns[20:26], *df.columns[43:49]]
olds = ['10대', '20대', '30대', '40대', '50대', '60대']
olds_eng = ["10's", "20's", "30's", "40's", "50's", "60's"]
items = [old+'의 매출금액' for old in olds]+[old+'의 건수별매출액' for old in olds]

# explain 7 : 연령대별
my_df = df.iloc[-4:][look]
for i in range(4):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
my_df.index = df.iloc[-4:]['ds']

# 설명하기
say = '2022년 각 분기당 연령대별 매출 금액은 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기: '
    for i, old in enumerate(olds):
        num = format(int(my_df[look[i]][qu]), ',')
        say += f'{old}는 {num}원, '
    say = say[:-2]
    say += '입니다.\n'

say += '2022년 각 분기당 연령대별 매출건수는 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기: '
    for i, old in enumerate(olds):
        num = format(int(my_df[look[i+6]][qu]), ',')
        say += f'{old}는 {num}건, '
    say = say[:-2]
    say += '입니다.\n'
print(say)
# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain7.txt', 'w') as f:
    f.write(say)

# 도표제작
fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))

fig1 = my_df[my_df.columns[:6]]
fig1.columns = olds_eng
fig1.index = df.iloc[-4:]['ds']


pl_fig = fig1.plot(
    kind='bar', ax=axes[0], title="Profit by generation", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = my_df[my_df.columns[6:]]
fig2.columns = olds_eng
fig2.index = df.iloc[-4:]['ds']

pl_fig2 = fig2.plot(
    kind='bar', ax=axes[1], title="Sales by generation", grid=True)
pl_fig2.get_yaxis().get_major_formatter().set_scientific(False)
plt.legend(loc=(1, 0.5))
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain7.png",
            dpi=300, bbox_inches='tight')

# explain 8 : 연령대별 추세
# 트렌드 구하기
# LIS & LDS
backup_a = []
trend = []
for index, elm in enumerate(look):
    if index < len(look)//2:
        for forecast in forecasts:
            if elm in forecast:
                break
        forecast_df = pd.read_csv(search_dir+forecast, index_col=0)

        a = list(forecast_df.iloc[-12:]['trend']/num_shops_df)
    else:
        a = list((df.iloc[-16:][look[index-2]] /
                  df.iloc[-16:][elm]).diff(periods=1)[-12:])

    backup_a.append(a)
#     print(elm, forecast, sum(a))
    dp_i = [0 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[i] > a[j] and dp_i[i] < dp_i[j]:
                dp_i[i] = dp_i[j]
        dp_i[i] += 1

    dp_d = [1 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[j] > a[i]:
                dp_d[i] = max(dp_d[i], dp_d[j]+1)

    if max(dp_d) >= 8 and max(dp_i) < 8:
        trend.append('-')
    elif max(dp_d) < 8 and max(dp_i) >= 8:
        trend.append('+')
    else:
        trend.append('=')

top_3 = list(pd.DataFrame(my_df.iloc[:, :6].sum()).sort_values(
    by=0, ascending=False).index[:3])
top_3_int = ', '.join([re.sub(r'[^0-9]', '', i) for i in top_3])
# 설명하기
say = f'매출금액과 건수대비 판매액을 연령별에 따라 분석하였으며, 매출액이 높은 연령대({top_3_int})는 상세분석을 추가로 진행하였습니다.\n'
state = []
for i, v in enumerate(trend):
    if v == '+':
        state.append('증가할')
    if v == '-':
        state.append('감소할')
    if v == '=':
        state.append('유지될')

for i, old in enumerate(olds):
    say += '\n'
    say += f'{items[i]}은 {state[i]} 것이며, {items[i+6]}은 {state[i+6]} 것으로 예상됩니다. '
    if not look[i] in top_3:
        continue
    say += f"즉 {old}에게의 인기가 현재 "
    if trend[i] == '+':
        say += '성장중에 있습니다. '
    elif trend[i] == '-':
        say += '침체중에 있습니다. '
    elif trend[i] == '=':
        say += '현상유지상태입니다. '

    if trend[i+6] == '+':
        say += '특정 상품의 가치가 높아지고 있음을 뜻합니다. 해당 상품을 고급화 혹은 브랜드화 하는 것을 추천합니다.'
    elif trend[i+6] == '-':
        say += '전체 상품의 평균 가치가 떨어지고 있음을 뜻합니다. 미끼상품같이 이윤이 적은 상품이 많이 팔리는 것으로 보이므로 해당 상품의 판매를 중지하거나 줄이는 것을 추천합니다.'
    elif trend[i+6] == '=':
        say += '매우 평범한 상태를 의미합니다. 지금은 다양한 판매전략을 세우고 시행하기 가장 좋은 시기이므로 많은 판매전략을 시도해보시는 것이 좋을 것으로 생각됩니다.'

#


print(say)

# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain8.txt', 'w') as f:
    f.write(say)

# 도표제작
fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
fig1 = pd.DataFrame(backup_a[:6]).T
fig1.columns = olds_eng
fig1.index = df.iloc[-12:]['ds']


pl_fig = fig1.plot(
    ax=axes[0], title="Profit trend by Generation", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = pd.DataFrame(backup_a[6:]).T
fig2.columns = olds_eng
fig2.index = df.iloc[-12:]['ds']

fig2.plot(ax=axes[1], title="Sales trend by Generation", grid=True)
plt.legend(loc=(1, 0.5))
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain8.png",
            dpi=300, bbox_inches='tight')

# 3. 향후 전망분석


# 불러온 데이터 중 분석할 데이터 정하기
# 분기당_매출_금액, 분기당_매출_건수, 점포수
look = [df.columns[1], df.columns[2], df.columns[-1]]
look_eng = ['Profit_all', 'Sales_all', 'Num of Stores']

# explain 9 : 수치 설명
my_df = df.iloc[-4:][look]
for i in range(2):
    my_df.iloc[:, i] //= df.iloc[-4:]['점포수']
my_df.index = df.iloc[-4:]['ds']
# 설명하기
say = '2022년 각 분기당 총 매출 금액은 다음과 같이 예상됩니다.\n'
for qu in df.iloc[-4:]['ds']:
    say += f'{qu}분기: '

    num = format(int(my_df[look[0]][qu]), ',')
    say += f'{look[0]}은 {num}원, '
    num = format(int(my_df[look[1]][qu]), ',')
    say += f'{look[1]}는 {num}건, '
    num = format(int(my_df[look[2]][qu]), ',')
    say += f'{look[2]}는 {num}개, '
    say = say[:-2]
    say += '입니다.\n'


print(say)
# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain9.txt', 'w') as f:
    f.write(say)

# 도표저장
fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 5))

fig1 = my_df[look[0]]
fig1.columns = look_eng[0]
fig1.index = df.iloc[-4:]['ds']


pl_fig = fig1.plot(
    kind='bar', ax=axes[0], title=f"{look_eng[0]}", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = my_df[look[1]]
fig2.columns = look_eng[1]
fig2.index = df.iloc[-4:]['ds']

pl_fig2 = fig2.plot(kind='bar', ax=axes[1], title=f"{look_eng[1]}", grid=True)
pl_fig2.get_yaxis().get_major_formatter().set_scientific(False)

fig3 = my_df[look[2]]
fig3.columns = look_eng[2]
fig3.index = df.iloc[-4:]['ds']

pl_fig3 = fig3.plot(kind='bar', ax=axes[2], title=f"{look_eng[2]}", grid=True)
pl_fig3.get_yaxis().get_major_formatter().set_scientific(False)
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain9.png",
            dpi=300, bbox_inches='tight')

# explain 10 : 전망 트렌드
# 트렌드 구하기
# LIS & LDS
backup_a = []
trend = []

for index, elm in enumerate(look):
    if index < len(look)-1:
        for forecast in forecasts:
            if elm in forecast:
                break
        forecast_df = pd.read_csv(search_dir+forecast, index_col=0)

        a = list(forecast_df.iloc[-12:]['trend']/num_shops_df)
    else:
        a = list(forecast_df.iloc[-12:]['trend'])

    backup_a.append(a)
#     print(elm, forecast, sum(a))
    dp_i = [0 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[i] > a[j] and dp_i[i] < dp_i[j]:
                dp_i[i] = dp_i[j]
        dp_i[i] += 1

    dp_d = [1 for i in range(len(a))]
    for i in range(len(a)):
        for j in range(i):
            if a[j] > a[i]:
                dp_d[i] = max(dp_d[i], dp_d[j]+1)

    if max(dp_d) >= 8 and max(dp_i) < 8:
        trend.append('-')
    elif max(dp_d) < 8 and max(dp_i) >= 8:
        trend.append('+')
    else:
        if max(dp_d) > max(dp_i):
            trend.append('-')
        else:
            trend.append('+')

# 트렌드를 바탕으로 결과설명하기
if trend == ['+', '+', '+']:
    say = '매출액과 매출건수, 점포수가 증가하는 추세입니다. 현재 해당 산업의 전망이 좋습니다! 하지만 이에 따라 경쟁 업체도 증가하므로 레드 오션 상태가 될 수 있으니 주의하시기 바랍니다.'
elif trend[0] == '+' and trend[2] == '-':
    say = '매출액은 증가하지만, 점포수는 감소하는 추세입니다. 현재 해당 산업의 전망이 좋습니다! 또한 아직 경쟁업체가 적거나 줄어드는 추세이므로, 빠르고 확실하게 준비하여 사업을 시작하는 것이 좋아보입니다.'
elif trend == ['-', '+', '+']:
    say = '매출건수와 점포수는 증가하지만 매출액은 감소하는 추세입니다. 현재 해당 산업은 레드 오션 상태이며, 출혈경쟁을 하고 있을 가능성이 매우 높습니다.'
elif trend == ['-', '-', '+']:
    say = '매출액과 매출건수는 감소하고 있으나, 점포수는 증가하는 추세입니다. 현재 해당 산업은 과잉상태입니다.'
elif trend[0] == '-' and trend[2] == '-':
    say = '매출액과 매출건수가 감소하고 있으며, 점포수도 감소하는 추세입니다. 현재 해당 산업의 전망이 좋지 않습니다.'

print(say)

# 결과 저장하기
# 파일저장
if not os.path.isdir(f'{root}/explain_long_term_work'):
    os.mkdir(f'{root}/explain_long_term_work')
if not os.path.isdir(f'{root}/explain_long_term_work/{work_argv}'):
    os.mkdir(f'{root}/explain_long_term_work/{work_argv}')
with open(f'{root}/explain_long_term_work/{work_argv}/explain10.txt', 'w') as f:
    f.write(say)

# 도표저장
fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 5))
fig1 = pd.DataFrame(backup_a[0])
fig1.index = df.iloc[-12:]['ds']


pl_fig = fig1.plot(
    ax=axes[0], title=f"{look_eng[0]}'s trend", grid=True, legend=None)
pl_fig.get_yaxis().get_major_formatter().set_scientific(False)

fig2 = pd.DataFrame(backup_a[1])
# fig2.columns =[look_eng[1]]
fig2.index = df.iloc[-12:]['ds']

fig2.plot(ax=axes[1], title=f"{look_eng[1]}'s trend", grid=True)

fig3 = pd.DataFrame(backup_a[2])
# fig3.columns =[look_eng[2]]
fig3.index = df.iloc[-12:]['ds']

fig3.plot(ax=axes[2], title=f"{look_eng[2]}'s trend", grid=True)

plt.legend(loc=(1, 0.5))
plt.savefig(f"{root}/explain_long_term_work/{work_argv}/explain10.png",
            dpi=300, bbox_inches='tight')
