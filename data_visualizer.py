from functools import lru_cache


def csv_to_pickle(csv_file: str, sep=";", pickle_file=None) -> None:
    """
    Converts a csv file to a pickle file
    :param csv_file: The csv file to convert
    :param sep: The separator used in the CSV file
    :param pickle_file: Filepath of the destination pickle file
    :return:
    """
    import pandas as pd
    if not pickle_file:
        pickle_file = csv_file[:-3] + "pickle"
        print("SAVED TO:", pickle_file)
    
    pd.read_csv(csv_file, sep=sep).to_pickle(pickle_file)

def visualize_profiles(directory: str, model) -> None:
    """
    Visualize the profiles at the given directory as tokens and output to a CSV file
    :param directory: The directory containing the profiles
    :param model: The model to use to convert ids to tokens, should be the same as the original tokenizer model
    :return:
    """
    import os
    from tqdm import tqdm
    import pandas as pd
    profile_files = os.listdir(directory)
    print("PROFILES:", profile_files)

    profiles = []

    for prf in tqdm(profile_files):
        profile = pd.read_csv(directory + "/" + prf, header=0, names=["token", "weight"])
        profile.loc[:, "token"] = model.convert_ids_to_tokens(profile.loc[:, "token"].values.tolist())
        profiles.append(profile)
        profile.to_csv("temp_data.csv")

def visualize_output(pickle_file: str) -> None:
    """
    Visualize the relatedness of the given output using UMAP dimensionality reduction
    :param pickle_file: The file containing the output of the model
    :return:
    """
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import umap
    sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    df = pd.read_pickle(pickle_file).sample(frac=0.01)
    reducer = umap.UMAP()
    data = df.iloc[:, 0].tolist()
    years = df.iloc[:, 1]
    del df
    scaled_data = StandardScaler().fit_transform(data)
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
def cmatrix_df(pickle_file: str):
    """
    Convert a long dataframe of confusion data to a confusion matrix
    :param pickle_file: The pickle file of the dataframe containing each pair of expected and estimated periods
    :return: The DataFrame of the confusion matrix
    """
    import pandas as pd
    from tqdm import tqdm
    data = pd.read_pickle(pickle_file)
    num_tests = len(data.index)
    data.sort_values(by=["estimate", "expected"], ignore_index=True, inplace=True)
    matrix = pd.DataFrame(0, columns=sorted(list(set(data["expected"].tolist()))),
                          index=sorted(list(set(data["expected"].tolist()))))

    for i, row in tqdm(data.iterrows(), total=len(data.index)):
        matrix.loc[row["expected"], row["estimate"]] += 1

    def to_ratio(freq: int) -> float:
        return round((freq / num_tests), 3)

    matrix = matrix.map(to_ratio)

    return matrix

def confusion_matrix(pickle_file: str, name: str) -> None:
    """
    Create a confusion matrix from a serialized long DataFrame of confusion data
    :param pickle_file: The pickle file of the dataframe containing each pair of expected and estimated periods
    :param name: The name of the model
    :return:
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    matrix = cmatrix_df(pickle_file)
    
    plt.figure(figsize=(12, 9))
    
    ax = sns.heatmap(matrix, cmap="viridis", square=True, cbar_kws={"label": "Ratio of Total Tests"}, vmin=0, vmax=0.03)
    # ax.set(xlabel="Estimated Time Periods", ylabel="Actual Time Periods")
    ax.set_xlabel("Predicted Time Periods", fontsize=10)
    ax.set_ylabel("Actual Time Periods", fontsize=10)
    ax.set_title(f"Confusion Matrix of {name} Model", fontsize=20)
    plt.savefig(f"{name}_cmatrix.png", dpi=1200, format="png")
    plt.show()

def confusion_matrices(pickle_files: dict) -> None:
    """
    Create confusion matrices from serialized long DataFrames of confusion data in a single image
    :param pickle_files: A dictionary of {name of test: DataFrame} pairs
    :return:
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def draw_heatmap(data, **kwargs):
        data = data.drop(columns="amt-removed")
        sns.heatmap(data, square=True, cbar=False, vmin=0, vmax=0.03, cmap="viridis", **kwargs)

    long_df = pd.DataFrame(columns=range(1800, 2010, 10))

    for name in pickle_files:
        matrix = cmatrix_df(pickle_files[name])
        matrix["amt-removed"] = [name] * len(matrix)

        long_df = pd.concat([long_df, matrix])

    g = sns.FacetGrid(long_df, col="amt-removed", sharey=False)
    g.map_dataframe(draw_heatmap)
    g.set_axis_labels("Predicted Time Periods", "Actual Time Periods")
    g.set_titles("{col_name}")

    sns.color_palette("viridis", as_cmap=True)

    plt.tight_layout()
    plt.savefig("heatmaps.png", dpi=4800)
    plt.show(dpi=2400)

def graph_results(pickle_file: str) -> None:
    """
    Create a bar graph of the results of the model using a serialized DataFrame of accuracy measurements
    :param pickle_file: The serialized DataFrame of accuracy measurements
    :return:
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    results = pd.read_pickle(pickle_file)
    
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
    
    fig = plt.figure()
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data, x="Amount Removed", y="Accuracy", hue="Metric", errorbar="sd", palette="viridis_r", err_kws={"linewidth": 1, "color": "#000000"}, capsize=0.25)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Accuracy of Lexical Extraction Models")
    ax.set_xticks(ax.get_xticks(), ["BASE", 0.0625, 0.125, 0.25, 0.5])
    fig.savefig('accs.png', bbox_inches='tight', dpi=2400)
    plt.show()

if __name__ == "__main__":
    # RUN FUNCTIONS TO VISUALIZE DATA
    pass