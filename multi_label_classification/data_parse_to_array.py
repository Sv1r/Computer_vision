from global_constants import *
from utils import create_tabular_data

df = create_tabular_data()
df.to_csv(f'{SAVE_METADATA}\\metadata.csv', sep=' ')
