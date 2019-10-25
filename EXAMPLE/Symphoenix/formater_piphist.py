import csv
from datetime import datetime
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
import zipfile

symbol = "USDCAD"
months = 10

# csv file name

rows = []

for i in range(1, months):
    zipfiles = dl(year='2019', month=i, pair=symbol, platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
    filename = "DAT_ASCII_%s_M1_20190%s" % (symbol, str(i))
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

with open('STATICS/FX/2019/%s.csv' % symbol, mode='w', newline='') as ticker:
    ticker_writer = csv.writer(ticker, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        ticker_writer.writerow(row)

csvfile.close()
ticker.close()
