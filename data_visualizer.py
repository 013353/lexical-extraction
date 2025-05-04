from functools import lru_cache


def csv_to_pickle(csv_file: str, sep=";", pickle_file=None):
    import pandas as pd
    if not pickle_file:
        pickle_file = csv_file[:-3] + "pickle"
        print(pickle_file)
    
    pd.read_csv(csv_file, sep=sep).to_pickle(pickle_file)

def visualize_profiles(model):
    import os
    from tqdm import tqdm
    import pandas as pd
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

def visualize_output(file: str):
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

@lru_cache
def cmatrix_df(file: str):
    import pandas as pd
    from tqdm import tqdm
    data = pd.read_pickle(file)
    num_tests = len(data.index)
    data.sort_values(by=["estimate", "expected"], ignore_index=True, inplace=True)
    matrix = pd.DataFrame(0, columns=sorted(list(set(data["expected"].tolist()))),
                          index=sorted(list(set(data["expected"].tolist()))))

    for i, row in tqdm(data.iterrows(), total=len(data.index)):
        matrix.loc[row["expected"], row["estimate"]] += 1

    def to_ratio(freq: int) -> float:
        return round((freq / num_tests), 3)

    matrix = matrix.map(to_ratio)

    print(matrix)
    return matrix

def confusion_matrix(file: str, name: str):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    matrix = cmatrix_df(file)
    
    plt.figure(figsize=(12, 9))
    
    ax = sns.heatmap(matrix, cmap="viridis", square=True, cbar_kws={"label": "Ratio of Total Tests"}, vmin=0, vmax=0.03)
    # ax.set(xlabel="Estimated Time Periods", ylabel="Actual Time Periods")
    ax.set_xlabel("Predicted Time Periods", fontsize=10)
    ax.set_ylabel("Actual Time Periods", fontsize=10)
    ax.set_title(f"Confusion Matrix of {name} Model", fontsize=20)
    plt.savefig(f"{name}_cmatrix.png", dpi=1200, format="png")
    plt.show()

def confusion_matrices(files: dict):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def draw_heatmap(data, **kwargs):
        data = data.drop(columns="amt-removed")
        sns.heatmap(data, square=True, cbar=False, vmin=0, vmax=0.03, cmap="viridis", **kwargs)

    long_df = pd.DataFrame(columns=range(1800, 2010, 10))

    for name in files:
        matrix = cmatrix_df(files[name])
        matrix["amt-removed"] = [name] * len(matrix)

        long_df = pd.concat([long_df, matrix])

    print(long_df)

    g = sns.FacetGrid(long_df, col="amt-removed", sharey=False)
    g.map_dataframe(draw_heatmap)
    g.set_axis_labels("Predicted Time Periods", "Actual Time Periods")
    g.set_titles("{col_name}")

    sns.color_palette("viridis", as_cmap=True)

    plt.tight_layout()
    plt.savefig("heatmaps.png", dpi=4800)
    plt.savefig("heatmaps.svg")
    plt.show(dpi=2400)

def graph_results(file: str):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    results = pd.read_csv(file)
    
    print(results)
    
    data = pd.DataFrame(columns=["Amount Removed", "Metric", "Accuracy"])
    
    cur_amt = 0
    for i, row in results.iterrows():
        for j, el in row.items():
            if j == "amt":
                cur_amt = el
            else:
                match j:
                    case "acc":
                        j = "Acc"
                    case "acc_3":
                        j = "Acc@3"
                    case "acc_5":
                        j = "Acc@5"
                data.loc[len(data)] = [cur_amt, j, el]

    print(data)
    
    fig = plt.figure()
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data, x="Amount Removed", y="Accuracy", hue="Metric", errorbar="sd", palette="viridis_r", err_kws={"linewidth": 1, "color": "#000000"}, capsize=0.25)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Accuracy of Lexical Extraction Models")
    ax.set_xticks(ax.get_xticks(), ["BASE", 0.0625, 0.125, 0.25, 0.5])
    plt.show()
    # fig.savefig('accs.png', bbox_inches='tight', dpi=2400)
    
def head(file: str):
    import pandas as pd
    
    pd.read_pickle(file).iloc[:10].to_csv("head.csv")

if __name__ == "__main__":
    pass