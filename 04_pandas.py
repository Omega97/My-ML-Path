"""use pandas to explore a dataset"""
from omar_utils.data.files import File  # https://github.com/Omega97/omar_utils
from omar_utils.basic.sets import Set


def load_data(url, path):
    """try to load data locally, else download"""
    return File(url=url, path=path, first_line_columns=True)()


def explore_data(data_frame):
    """explore data"""
    print('\n\n\tColumns\n')
    columns = data_frame.columns.values
    for i_ in columns:
        print(i_)


def elements(data_frame, name, max_elements=20):
    """:returns list of elements in Series if there are <= max_elements, else None"""
    s = Set()
    for i in data_frame[name]:
        s += Set(i)
        if len(s) > max_elements:
            return None
    return sorted(list(s))


def display_elements(data_frame, max_elements=20):
    """print elements of each Series in data_frame if < max_elements"""
    print('\n\n\tElements')
    for i in data_frame.columns.values:
        e = elements(data_frame, i, max_elements=max_elements)
        if e:
            print('\n')
            print(i)
            for j in e:
                print('-', j)


if __name__ == '__main__':

    # load data
    URL = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
    PATH = 'datasets/housing_data.txt'
    df = load_data(url=URL, path=PATH)

    # explore data
    explore_data(df)
    display_elements(df)
