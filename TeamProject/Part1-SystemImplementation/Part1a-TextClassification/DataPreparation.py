import pandas as pd

# Dataset has 15 classes (dialog acts)

# We want to read the dalog_acts.dat file. The first word of every line is the class label and the rest of the line is the text.
# Change accordingly, my computer does not work for relative paths
def get_data(path = 'dialog_acts',shuffel = True):

    df = pd.read_csv('dialog_acts.dat', names=['data'])

    if shuffel:
        df = df.sample(frac=1).reset_index(drop=True)

    # Apply the function to split the 'data' column into 'label' and 'text' columns
    df[['label', 'text']] = df['data'].apply(lambda x: pd.Series(x.split(' ', 1)))

    # Drop the original 'data' column
    df.drop('data', axis=1, inplace=True)

    # print(df.head(10))
    df_deduplicated = df.drop_duplicates(subset=['text'])

    # Features and Labels
    x = df['text']
    y = df['label']

    x_deduplicated = df_deduplicated['text']
    y_deduplicated = df_deduplicated['label']

    return x,y,x_deduplicated,y_deduplicated