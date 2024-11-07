import subprocess
import pandas as pd
import tqdm as tqdm
from tqdm import tqdm

models = pd.read_csv('models.csv')

n_layers = [1, 4, 16]
hidden_sizes = [64, 128, 256]
print(hidden_sizes)

type_table = {
    'gcn': 'mutigcn',
    'jknet': 'densegcn',
    'jknet_nodrop': 'densegcn',
    'gcn_nodrop': 'mutigcn',
}

for index, row in tqdm(models.iterrows(), total=models.shape[0], desc='Processing models'):
    model = row['model']
    dataset = row['dataset']
    lr = row['lr']
    weight_decay = row['weight_decay']
    sampling_percent = row['sampling_percent']
    dropout = row['dropout']
    normalization = row['normalization']
    withloop = row['withloop']
    withbn = row['withbn']
    nbaseblocklayer = row['nbaseblocklayer']

    for n_layer in n_layers:
        for hidden in hidden_sizes:
            #print(hidden)
            
            cmd = [
                'python', './src/train_new.py',
                '--debug',
                '--datapath', 'data//',
                '--seed', '42',
                '--dataset', dataset.lower(),
                '--type', type_table[model.lower()],
                '--nhiddenlayer', str(n_layer),
                '--nbaseblocklayer', str(nbaseblocklayer),
                '--hidden', str(hidden),
                '--epoch', '400',
                '--lr', str(lr),
                '--weight_decay', str(weight_decay),
                '--early_stopping', '400',
                '--sampling_percent', str(sampling_percent),
                '--dropout', str(dropout),
                '--normalization', normalization,
            ]

            if str(withloop).lower() == 'true':
                cmd.append('--withloop')
            if str(withbn).lower() == 'true':
                cmd.append('--withbn')

            print(" ".join(cmd).center(80, '-'))
            results = subprocess.run(cmd, capture_output=True, text=True)
            output = results.stdout
            error = results.stderr
            
            print(output)
            print(error)
            print('-' * 80)
        