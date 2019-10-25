import pandas as pd
import os
import io
import yaml
from typing import NamedTuple


def save_csv(df: pd.DataFrame, ticker: str, path: str):
    file_path = os.path.join(path, ticker + '.csv').replace('\\', '/')
    df.to_csv(file_path, index=False)


def write_yaml(data, file_path, encoding='uft8'):
    with io.open(file_path, 'w', encoding=encoding) as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def read_yaml(file_path, loader=yaml.SafeLoader):
    with open(file_path, 'r') as stream:
        output = yaml.load(stream, loader)
    return output


class Record(NamedTuple):
    """ Define the fields and their types in a record. """
    n_hist: int
    n_forward: int
    trade_th: float
    stop_loss: float
    cl: float

    @classmethod
    def transform(cls: 'Record', dct: dict) -> dict:
        """ Convert string values in given dictionary to corresponding Record
            field type.
        """
        return {field: cls._field_types[field](value)
                for field, value in dct.items()}
