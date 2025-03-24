def visualize_profiles(model):
    import os
    from tqdm import tqdm
    import pandas as pd
    from transformers import BertTokenizerFast
    from matplotlib import pyplot as plt
    profile_files = os.listdir("FR_Test/profiles")
    print("PROFILES:", profile_files)

    profiles = []

    for prf in tqdm(profile_files):
        profile = pd.read_csv("FR_Test/profiles/" + prf, header=0, names=["token", "weight"])
        print(profile)
        input()
        profile.loc[:, "token"] = model.convert_ids_to_tokens(profile.loc[:, "token"].values.tolist())
        profiles.append(profile)
        print(profile)
        profile.to_csv("temp_data.csv")
    
    # for profile in profiles:
    #     print(profile)

def visualize_output(file: str):
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import umap
    sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    df = pd.read_pickle(file).sample(frac=0.01)
    print(df)
    reducer = umap.UMAP()
    data = df.iloc[:, 0].tolist()
    years = df.iloc[:, 1]
    del df
    print(1)
    # print(data)
    scaled_data = StandardScaler().fit_transform(data)
    print(2)
    del data
    embedding = reducer.fit_transform(scaled_data)
    print(embedding.shape)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=years,
        cmap="summer"
    )
    plt.colorbar()
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of outputs', fontsize=24)
    plt.savefig("output.png")
    plt.show()

if __name__ == "__main__":
    visualize_output("FR_Test/train_outputs.csv")