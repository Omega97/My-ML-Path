import quandl
from omar_utils.data.files import File  # https://github.com/Omega97/omar_utils
from omar_utils.basic.plots import simple_plot

def load_data(path, name):
    try:
        # try to load data locally
        return File(path=path)()
    except FileNotFoundError:
        # download data from web
        df = quandl.get(name)
        f = File()
        f.load_var(df)
        f.save(path)
        return df


if __name__ == '__main__':

    # load data
    Df = load_data(path='datasets/GOOGL', name='WIKI/GOOGL')
    # print(Df)
    # print(Df.columns.values)

    # clean data
    Df['HL_PCT'] = (Df['Adj. High'] - Df['Adj. Low']) / Df['Adj. Close']    # normalized high-low ratio
    Df['change_ratio'] = (Df['Adj. Close'] - Df['Adj. Open']) / Df['Adj. Open']    # daily change ratio
    Df = Df[['Adj. Close', 'HL_PCT', 'change_ratio', 'Adj. Volume']]    # the only features we care about
    print(Df)

    simple_plot(list(Df['Adj. Close']))
