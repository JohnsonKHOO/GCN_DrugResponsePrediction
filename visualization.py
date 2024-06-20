from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import math


def visualize(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['mol_weight'], bins=50, kde=True)
    plt.title('Distribution of Molecular Weights')
    plt.xlabel('Molecular Weight')
    plt.ylabel('Frequency')
    plt.savefig('./dataset/visualization/molecular_weight_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['IC50 (nM)'], bins=50, kde=True)
    plt.title('Distribution of IC50 Values')
    plt.xlabel('IC50 (nM)')
    plt.ylabel('Frequency')
    plt.savefig('./dataset/visualization/ic50_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mol_weight', y='IC50 (nM)', data=df)
    plt.title('Molecular Weight vs. IC50')
    plt.xlabel('Molecular Weight')
    plt.ylabel('IC50 (nM)')
    plt.savefig('./dataset/visualization/mol_weight_vs_ic50.png')
    plt.show()


def drawMols(df, batch_size):
    mols = [Chem.MolFromSmiles(smile) for smile in df['smiles']]
    legends = df['mol_name'].tolist()

    if batch_size < 100:
        img = Draw.MolsToGridImage(mols[:batch_size], molsPerRow=5, subImgSize=(500, 500), legends=legends)
        img.save('./dataset/visualization/ChemDraw/molecule_grid.png')
        img.show()
        return

    num_batches = math.ceil(len(mols) / batch_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(mols))

        batch_mols = mols[start_idx:end_idx]
        batch_legends = legends[start_idx:end_idx]

        img = Draw.MolsToGridImage(batch_mols, molsPerRow=5, subImgSize=(500, 500), legends=batch_legends)
        img.save(f'./dataset/visualization/ChemDraw/molecule_grid_{batch_idx + 1}.png')


def main():
    # df_ori = pd.read_csv(f'./dataset/molecule_data_ori.csv')
    # df_cleaned = pd.read_csv(f'./dataset/preprocess/molecule_data_cleaned.csv')
    df_scaled = pd.read_csv(f'./dataset/preprocess/molecule_data_scaled.csv')
    # visualize(df_ori)
    # visualize(df_scaled)
    drawMols(df_scaled, 25)


if __name__ == '__main__':
    main()
