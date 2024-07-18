import pandas as pd
import numpy as np

def prepare_data(df):
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values in categorical columns by sampling from existing values
    prev_ownership_values = df['Prev_ownership'].dropna().values
    if len(prev_ownership_values) > 0:
        df.loc[pd.isnull(df['Prev_ownership']), 'Prev_ownership'] = np.random.choice(prev_ownership_values, pd.isnull(df['Prev_ownership']).sum())
    else:
        df.loc[pd.isnull(df['Prev_ownership']), 'Prev_ownership'] = 'Unknown'

    curr_ownership_values = df['Curr_ownership'].dropna().values
    if len(curr_ownership_values) > 0:
        df.loc[pd.isnull(df['Curr_ownership']), 'Curr_ownership'] = np.random.choice(curr_ownership_values, pd.isnull(df['Curr_ownership']).sum())
    else:
        df.loc[pd.isnull(df['Curr_ownership']), 'Curr_ownership'] = 'Unknown'

    color_values = df['Color'].dropna().values
    if len(color_values) > 0:
        df.loc[pd.isnull(df['Color']), 'Color'] = np.random.choice(color_values, pd.isnull(df['Color']).sum())
    else:
        df.loc[pd.isnull(df['Color']), 'Color'] = 'Unknown'

    # Convert 'Test' column to the number of days since the test date
    df.loc[:, 'Test'] = pd.to_datetime(df['Test'], errors='coerce')
    df.loc[:, 'Test'] = (pd.Timestamp.now() - df['Test']).dt.days
    Test_values = df['Test'].dropna().values
    if len(Test_values) > 0:
        df.loc[pd.isnull(df['Test']), 'Test'] = np.random.choice(Test_values, pd.isnull(df['Test']).sum())
    else:
        df.loc[pd.isnull(df['Test']), 'Test'] = 0

    # Handle missing values in other columns
    df.loc[:, 'Gear'] = df['Gear'].fillna(df['Gear'].mode()[0] if not df['Gear'].mode().empty else 'Unknown')

    # Remove commas and convert numeric values
    df.loc[:, 'capacity_Engine'] = df['capacity_Engine'].replace(',', '', regex=True)
    df.loc[:, 'capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')  # Handle non-numeric values

    # Convert invalid values in the 'Km' column
    df.loc[:, 'Km'] = df['Km'].replace(',', '', regex=True)
    df.loc[:, 'Km'] = pd.to_numeric(df['Km'], errors='coerce')

    # Handle missing values using groupby and fill with median or mean
    df.loc[:, 'capacity_Engine'] = df.groupby(['manufactor', 'model'])['capacity_Engine'].transform(lambda x: x.fillna(x.median()))
    df.loc[:, 'Engine_type'] = df.groupby('model')['Engine_type'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown'))
    df.loc[:, 'Area'] = df['Area'].fillna(df['Area'].mode()[0] if not df['Area'].mode().empty else 'Unknown')
    df.loc[:, 'Pic_num'] = df['Pic_num'].fillna(df['Pic_num'].median())
    df.loc[:, 'Km'] = df.groupby('Year')['Km'].transform(lambda x: x.fillna(x.median()))
    df.loc[:, 'Supply_score'] = df.groupby('manufactor')['Supply_score'].transform(lambda x: x.fillna(x.median()))

    # Convert categorical columns to categories
    df.loc[:, 'manufactor'] = df['manufactor'].astype('category')
    df.loc[:, 'model'] = df['model'].astype('category')
    df.loc[:, 'Gear'] = df['Gear'].astype('category')
    df.loc[:, 'Engine_type'] = df['Engine_type'].astype('category')
    df.loc[:, 'Area'] = df['Area'].astype('category')
    df.loc[:, 'City'] = df['City'].astype('category')
    df.loc[:, 'Color'] = df['Color'].astype('category')
    df.loc[:, 'Prev_ownership'] = df['Prev_ownership'].astype('category')
    df.loc[:, 'Curr_ownership'] = df['Curr_ownership'].astype('category')

    # Remove outliers by calculating values outside the interquartile range (IQR)
    numeric_columns = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Test', 'Supply_score', 'Pic_num']
    Q1 = df[numeric_columns].quantile(0.15)
    Q3 = df[numeric_columns].quantile(0.85)
    IQR = Q3 - Q1
    df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df