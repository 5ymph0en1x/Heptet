import csv
from datetime import datetime
import pandas as pd
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
import zipfile

###############################
symbol = "USDCAD"
from_year = 2018
until_year = 2019
until_month = 10
###############################

rows = []
current_year = int(datetime.now().year)
current_month = int(datetime.now().month)

if until_year > current_year:
    until_year = current_year

if from_year > current_year:
    from_year = current_year

if until_month > current_month or until_month > 12:
    until_month = current_month

for j in range(from_year, until_year + 1):
    if j != current_year:
        zipfiles = dl(year=j, month=None, pair=symbol, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
        filename = "DAT_ASCII_%s_M1_%s" % (symbol, str(j))
        zip_filename = "./" + filename + ".zip"
        csv_filename = filename + ".csv"
        with zipfile.ZipFile(zip_filename) as z:
            z.extract(csv_filename)
            with open(csv_filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';')
                # extracting each data row one by one
                for row in csvreader:
                    # print(row[0])
                    row_time = datetime.strptime(row[0], '%Y%m%d %H%M%S')
                    row_ = row_time, row[4]
                    rows.append(row_)
        continue
    for i in range(1, until_month + 1):
        zipfiles = dl(year=j, month=i, pair=symbol, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
        if i < 10:
            filename = "DAT_ASCII_%s_M1_%s0%s" % (symbol, str(j), str(i))
        if i >= 10:
            filename = "DAT_ASCII_%s_M1_%s%s" % (symbol, str(j), str(i))
        zip_filename = "./" + filename + ".zip"
        csv_filename = filename + ".csv"
        with zipfile.ZipFile(zip_filename) as z:
            z.extract(csv_filename)
            with open(csv_filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';')
                # extracting each data row one by one
                for row in csvreader:
                    # print(row[0])
                    row_time = datetime.strptime(row[0], '%Y%m%d %H%M%S')
                    row_ = row_time, row[4]
                    rows.append(row_)

time_start = rows[0][0]
time_end = rows[len(rows) - 1][0]
index = pd.date_range(time_start, time_end, freq='min')
ts = pd.DataFrame(rows, columns=['date', 'close'])
ts.set_index('date', drop=True, inplace=True)
ts = ts.reindex(index=index, method='pad')
ts = ts[ts.index.dayofweek < 5]
ts.to_csv(path_or_buf='STATICS/FX/%s.csv' % symbol, index=True, index_label='date')

csvfile.close()
