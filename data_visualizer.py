import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizerFast
from matplotlib import pyplot as plt

def visualize_profiles(model):
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

if __name__ == "__main__":
    bert = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    visualize_profiles(bert)