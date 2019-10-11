import quandl
from omar_utils.data.files import File  # https://github.com/Omega97/omar_utils


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
    print(Df)

    # clean data
    print(Df['Open'])
