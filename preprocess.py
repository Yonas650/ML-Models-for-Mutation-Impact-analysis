import pandas as pd
from Bio import SeqIO

#paths to data files
brca1_fasta_file = "data/P38398.fasta"
brca1_txt_file = "data/P38398.txt"
brca2_fasta_file = "data/P51587.fasta"
brca2_txt_file = "data/P51587.txt"

#function to read FASTA files
def read_fasta(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return record.id, str(record.seq)

#function to read TXT annotation files
def read_annotation(txt_file):
    with open(txt_file, 'r') as file:
        data = file.readlines()
    return data

#read BRCA1 protein sequence
brca1_id, brca1_seq = read_fasta(brca1_fasta_file)
print(f"BRCA1 ID: {brca1_id}")
print(f"BRCA1 Sequence (first 100 amino acids): {brca1_seq[:100]}")

#read BRCA2 protein sequence
brca2_id, brca2_seq = read_fasta(brca2_fasta_file)
print(f"BRCA2 ID: {brca2_id}")
print(f"BRCA2 Sequence (first 100 amino acids): {brca2_seq[:100]}")

#read BRCA1 annotations
brca1_annotations = read_annotation(brca1_txt_file)
print(f"BRCA1 Annotations: {brca1_annotations[:10]}")

#read BRCA2 annotations
brca2_annotations = read_annotation(brca2_txt_file)
print(f"BRCA2 Annotations: {brca2_annotations[:10]}")

#save protein sequences to a CSV file for further analysis
protein_data = {
    'Protein_ID': [brca1_id, brca2_id],
    'Protein_Sequence': [brca1_seq, brca2_seq]
}

df_proteins = pd.DataFrame(protein_data)
df_proteins.to_csv('data/brca_protein_sequences.csv', index=False)
print("Protein sequences saved to 'data/brca_protein_sequences.csv'")

#save annotations to text files for detailed analysis
with open('data/brca1_annotations.txt', 'w') as f:
    f.writelines(brca1_annotations)
print("BRCA1 annotations saved to 'data/brca1_annotations.txt'")

with open('data/brca2_annotations.txt', 'w') as f:
    f.writelines(brca2_annotations)
print("BRCA2 annotations saved to 'data/brca2_annotations.txt'")
