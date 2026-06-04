import datetime

import chinese_calendar as ccl
import cnlunar as clu

new_year = datetime.date(2026, 2, 17)

# 判断是否是工作日
cet = ccl.is_workday(new_year)
print(cet)
# 会输出 False


# 判断工作日的时候同时输出节日名
det = ccl.get_holiday_detail(new_year)
print(det)
# 会输出 (True, 'Spring Festival')

# 判断农历（这里用的是大年初一的日期）
chu_yi = clu.Lunar(datetime.datetime(2026, 2, 17, 10, 30), godType='8char')
print(chu_yi.lunarMonth)
print(chu_yi.lunarDay)
print(chu_yi.weekDayCn)
# 会输出 1 和 1 和 星期二，意味着2026年2月17日代表农历1月1日，星期二