import argparse
import pandas as pd

args = None
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--input_path', dest='path', type=str, help='Path to the input PDB file.')
parser.add_argument('--df_path', dest='df_path', type=str, help='Path to the database.')
parser.add_argument('--output_path', dest='output_path', type=str, help='Path to save the output file.')

arguments = parser.parse_args()

def main(args):

    h_seq = []
    l_seq = []
    h_pos = 11
    l_pos = 11
    offset = 8

    amino_acid_dictionary = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    df = pd.read_csv(args.df_path, sep='\t', header=0)[['pdb', 'Hchain', 'Lchain', 'compound']]
    df.drop_duplicates(keep='first', subset='pdb', inplace=True)
    df = df[df['pdb'] == args.path[-8:-4]]
    heavy = df['Hchain'].item()
    light = df['Lchain'].item()
    pdb = df['pdb'].item()
    compound = df['compound'].item()

    with open(args.path, 'r') as f: 
        for line in f:
            line = line.strip()
            if line.find('SEQRES') != -1:
                if line[h_pos] == heavy:
                    for i in range(0, len(line), 2):
                        pos = h_pos + offset + i
                        amino = line[pos:pos+3]
                        if amino != '':
                            line = line.replace(amino, amino_acid_dictionary[amino], 1)
                    h_seq.append(line[h_pos+offset:].replace(' ', ''))
                elif line[l_pos] == light:
                    for i in range(0, len(line), 2):
                        pos = l_pos + offset + i
                        amino = line[pos:pos+3]
                        if amino != '':
                            line = line.replace(amino, amino_acid_dictionary[amino], 1)
                    l_seq.append(line[l_pos+offset:].replace(' ', ''))
    
    h_descr = pdb + ' ' + compound + '. HEAVY CHAIN'
    l_descr = pdb + ' ' + compound + '. LIGHT CHAIN'
    h_seq = ''.join(h_seq)
    l_seq = ''.join(l_seq)

<<<<<<< HEAD
    with open(args.output_path + pdb + '_h.fasta', "w") as f_new:
        f_new.writelines('>' + h_descr + '\n')
        f_new.writelines(h_seq + '\n')

    with open(args.output_path + pdb + '_l.fasta', "w") as f_new:
=======
    with open(args.output_path + 'sabdab_h.fasta', "a") as f_new:
        f_new.writelines('>' + h_descr + '\n')
        f_new.writelines(h_seq + '\n')

    with open(args.output_path + 'sabdab_l.fasta', "a") as f_new:
>>>>>>> master
        f_new.writelines('>' + l_descr + '\n')
        f_new.writelines(l_seq + '\n')

if __name__ == '__main__':
    main(arguments)
