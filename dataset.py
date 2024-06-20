import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from concurrent.futures import ThreadPoolExecutor


def process_molecule(mol, i):
    if mol is not None:
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) == -1:
            print(f'Embedding failed for molecule {i}')
            return None
        mol_id = mol.GetProp('_Name')
        smiles = Chem.MolToSmiles(mol)
        mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        ic_50 = mol.GetProp('IC50 (nM)')
        return {'mol_name': mol_id, 'smiles': smiles, 'mol_weight': mol_weight, 'IC50 (nM)': ic_50}
    return None


def process_chunk(chunk, chunk_index):
    data = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_molecule, mol, i) for i, mol in enumerate(chunk)]
        for future in futures:
            result = future.result()
            if result is not None:
                data.append(result)
    df = pd.DataFrame(data)
    df.to_csv(f'./dataset/ThreadPool/molecule_data_chunk_{chunk_index}.csv', index=False)
    return len(data)


def main():
    suppl = Chem.SDMolSupplier('./dataset/BindingDB_PubChem_3D.sdf')
    chunk_size = 1000  # Adjust chunk size
    chunk = []
    chunk_index = 0
    total_processed = 0

    start_time = time.time()

    for i, mol in enumerate(suppl):
        chunk.append(mol)
        if (i + 1) % chunk_size == 0:
            total_processed += process_chunk(chunk, chunk_index)
            chunk = []
            chunk_index += 1

    if chunk:
        total_processed += process_chunk(chunk, chunk_index)

    end_time = time.time()
    print(f"Processed {total_processed} molecules in {end_time - start_time:.2f} seconds")

    # Combine all chunks into a single DataFrame
    all_data = []
    for i in range(chunk_index + 1):
        df_chunk = pd.read_csv(f'./dataset/ThreadPool/molecule_data_chunk_{i}.csv')
        all_data.append(df_chunk)

    df = pd.concat(all_data, ignore_index=True)
    df.to_csv('./dataset/molecule_data_ori.csv', index=False)
    print("Combined DataFrame created and saved to 'molecule_data_combined.csv'")


if __name__ == '__main__':
    main()
