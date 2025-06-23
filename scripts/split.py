import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    input_csv = "../data/dataset.csv"
    df = pd.read_csv(input_csv)
    
    df['CategoriesList'] = (
        df['Categories']
        .fillna('')
        .str.split(r'\s*[\n,]\s*', regex=True)
        .apply(lambda lst: [s for s in lst if s])
    )

    df_pairs = (
        df
        .explode('CategoriesList')
        [['Title','CategoriesList']]
        .rename(columns={'Title':'anchor',
                         'CategoriesList':'positive'})
    )

    test_size = 0.1   
    val_size  = 0.3   
    seed      = 42

    anchors = df_pairs['anchor'].unique()

    remain_anchors, test_anchors = train_test_split(
        anchors,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    rel_val_size = val_size / (1 - test_size)

    train_anchors, val_anchors = train_test_split(
        remain_anchors,
        test_size=rel_val_size,
        random_state=seed,
        shuffle=True
    )

    train_df = df_pairs[df_pairs['anchor'].isin(train_anchors)]
    val_df   = df_pairs[df_pairs['anchor'].isin(val_anchors)]
    test_df  = df_pairs[df_pairs['anchor'].isin(test_anchors)]

    train_df.to_csv("../data/train_val_split/train.csv", index=False)
    val_df.to_csv(  "../data/train_val_split/val.csv",   index=False)
    test_df.to_csv( "../data/train_val_split/test.csv",  index=False)

    print(f"Saved {len(train_df)} rows to train.csv ({len(train_anchors)} titles)")
    print(f"Saved {len(val_df)}   rows to val.csv   ({len(val_anchors)} titles)")
    print(f"Saved {len(test_df)}  rows to test.csv  ({len(test_anchors)} titles)")

if __name__ == "__main__":
    main()
