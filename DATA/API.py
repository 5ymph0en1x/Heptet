import sys
import pandas as pd
from UTIL import AsyncIO, DateTime


def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)


class Tiingo(object):

    def __init__(self, config):
        self.config = config

    def fetch(self, ticker_list, attr):
        data_type = self.config['DataAPIDataType']

        if data_type == 'realtime':
            return self._fetch_realtime(ticker_list, attr)

        if (data_type == 'intraday') or (data_type == 'daily'):
            if self.config['DataAPIFetchMethod'] == 'async':
                return self._fetch_hist_async(ticker_list, attr)
            else:
                return self._fetch_hist(ticker_list, attr)

    def _fetch_realtime(self, ticker_list, attr):
        token   = self.config['DataAPIToken']
        tickers = ','.join(ticker_list)
        para    = self.get_url_realtime(tickers, token)
        output  = self.fetch_data(para)[attr]
        return output

    def _fetch_hist_async(self, ticker_list, attr):
        output = dict()
        for ticker in ticker_list:
            df      = pd.DataFrame()
            paras   = self._get_para(ticker)
            loop    = AsyncIO.create_loop()
            tasks   = AsyncIO.create_tasks(loop, paras, self.fetch_data_async)
            results = loop.run_until_complete(tasks)
            loop.close()
            for result in results:
                data = result.result()
                if len(data) != 0:
                    data = self.format_data(data, attr, self.config['DataAPINumberOfObsPerDay'])
                    df = pd.concat([df, data])
            df.reset_index(drop=True, inplace=True)
            output[ticker] = df
        return output

    def _fetch_hist(self, ticker_list, attr):
        output = dict()
        for ticker in ticker_list:
            df     = pd.DataFrame()
            paras  = self._get_para(ticker)
            outs   = self.fetch_data(paras)
            if len(outs) != 0:
                data = self.format_data(outs, attr, self.config['DataAPINumberOfObsPerDay'])
                df = pd.concat([df, data])
            df.reset_index(drop=True, inplace=True)
            output[ticker] = df
        return output

    def _get_para(self, ticker):
        config     = self.config
        data_type  = config['DataAPIDataType']
        token      = config['DataAPIToken']
        freq       = config['DataAPIFreq']
        start_date = config['DataAPIStartDate']
        end_date   = config['DataAPIEndDate']
        # period     = DateTime.get_dates_weekday(start_date, end_date)

        if data_type == 'intraday':
            paras = self.get_url_intraday(ticker, start_date, end_date, freq, token)
        elif data_type == 'daily':
            paras = [{'url': self.get_url_daily(ticker, start_date, end_date, token)}]
        else:
            raise ValueError('Error:Invalid data type.')

        print(paras)

        return paras

    @staticmethod
    def get_url_intraday(ticker, start_date, end_date, freq, token):
        url = 'https://api.tiingo.com/tiingo/fx/{ticker}/prices?startDate={start_date}' \
              '&endDate={end_date}&resampleFreq={freq}&token={token}' \
            .format(ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    freq=freq,
                    token=token)
        return url

    @staticmethod
    def get_url_realtime(ticker, token):
        url = 'https://api.tiingo.com/iex/?tickers={ticker}&token={token}'.format(ticker=ticker,
                                                                                  token=token)
        return url

    @staticmethod
    def get_url_daily(ticker, start_date, end_date, token):
        url = 'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}' \
              '&endDate={end_date}&token={token}' \
            .format(ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    token=token)
        return url

    @staticmethod
    async def fetch_data_async(session, url):
        async with session.get(url) as response:
            json = await response.json()
            try:
                data = pd.DataFrame(json)
            except ValueError:
                data = json
            return data

    @staticmethod
    def fetch_data(url):
        data = pd.read_json(url)
        # print(data)
        return data

    @staticmethod
    def format_data(data, attr, n_obs):
        n_obs = int(n_obs)
        data  = data[attr]
        if len(data) > n_obs:
            data = data.iloc[:n_obs]
        return data
