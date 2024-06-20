def normalise_participant_data(df):
    columns = df.columns
    skip_columns = [('df_labels', 'label'), ('df_labels', 'participant_id')]
    columns = [col for col in columns if col not in skip_columns]

    # normalize columns to interval [0, 1]
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    return df
