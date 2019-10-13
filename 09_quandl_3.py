from pandas import Series, DataFrame
import quandl
from omar_utils.data.files import File  # https://github.com/Omega97/omar_utils
from omar_utils.basic.plots import simple_plot
# from math import ceil


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


def clean_data(df: DataFrame):
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close']  # normalized high-low ratio
    # input()
    df['change_ratio'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']  # daily change ratio
    # input()
    out = df[['Adj. Close', 'HL_PCT', 'change_ratio', 'Adj. Volume']]  # the only features we care about
    # input()
    # out.fillna(-99999, inplace=True)
    return out


def chunk(s: Series, first=0, last=None) -> Series:
    """cut the series from first (included) to last (excluded)"""
    last = len(s) if last is None else last
    return s[first: last]


def predict(f, s: Series, length) -> Series:
    """use f on each chunk of length 'length' to predict next row"""
    out = Series([None for _ in range(len(s))])
    for i in range(len(s) - length):
        out[i+length] = f(*list(chunk(s, i, i+length)))
    return out


def loss(s1: Series, s2: Series):
    """standard deviation between s1 and s2"""
    length = min(len(s1), len(s2))
    dif = [(s1[i] - s2[i]) if s1[i] is not None and s2[i] is not None else 0 for i in range(length)]
    return (sum([i ** 2 for i in dif]) / (length - 1)) ** (1/2)


if __name__ == '__main__':

    # load data
    Df = load_data(path='datasets/GOOGL', name='WIKI/GOOGL')

    # clean data
    Df = clean_data(Df)

    # make predictions
    forecast_col = 'Adj. Close'  # predict this column


    def fit(a2, a1):
        return a1 + (a1 - a2)


    Pre = predict(fit, Df[forecast_col], 2)
    Pre.index = Df.index
    Df['label'] = Pre

    print('\nLoss =', loss(Df[forecast_col], Df['label']))

    simple_plot(list((Df[forecast_col] - Df['label']).dropna()))
