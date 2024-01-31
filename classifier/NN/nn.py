#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env/bin/python3.8
# coding: utf-8
#SBATCH -J ann
#SBATCH -o ann.log
#SBATCH -e ann.err
#SBATCH --mem=40G
#SBATCH --partition=gpu --gpus=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00

# In[11]:


import torch
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# In[5]:


local_stringdb = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/data/string/lfs-stringdb/'
# load local STRING database and names
stringdf = pd.read_csv(local_stringdb+'9606.protein.info.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
stringdf['preferred_name'] = stringdf['preferred_name'].str.upper()
stringId2name = stringdf.set_index('#string_protein_id')['preferred_name'].to_dict()
name2stringId = stringdf.set_index('preferred_name')['#string_protein_id'].to_dict()
stringdf = pd.read_csv(local_stringdb+'9606.protein.aliases.v12.0.txt', sep='\t', header=0, usecols=['#string_protein_id', 'alias']).drop_duplicates(['alias'], keep='first')
stringdf['alias'] = stringdf['alias'].str.upper()
aliases2stringId = stringdf.set_index('alias')['#string_protein_id'].to_dict()


# In[6]:


def get_df(dataset):

    files_path = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/results/models/Dataset-'+dataset+'--RE-True--W-True--Epochs-15000--D-4--RH-25--LR-0.1--LP-False--CUDA-True/'
    path1 = files_path + 'latent.pkl'
    path2 = files_path + 'RE.pkl'

    with open(path1, 'rb') as file:
        latent = pickle.load(file)
    latent = np.array(latent)
    with open(path2, 'rb') as file:
        re = pickle.load(file)
    re = np.array(re)

    convertpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/datasets/'+dataset+'/ppi_index.pkl'
    with open(convertpath, 'rb') as file:
        ppi_index = pickle.load(file)
    data = np.concatenate((latent, re[:, np.newaxis]), axis=1)
    df_latent = pd.DataFrame()
    for i, d in enumerate(range(data.shape[1])):
        if i == len(range(data.shape[1]))-1:
            col_name = 're'
        else:
            col_name = str(d+1)+'d'
        df_latent[col_name] = data.T[d]
    df_latent['node'] = df_latent.index
    inv_dict = {v: k for k, v in ppi_index.items()}
    df_latent = df_latent.add_prefix(dataset+'_')
    df_latent['gene'] = df_latent[dataset+'_node'].map(inv_dict)
    df = df_latent.loc[:, ~df_latent.columns.str.endswith('node')]
    return df


# In[7]:


ppidf = get_df('ppi')
scdf = get_df('sc')
stdf = get_df('st')
df = pd.merge(ppidf,scdf,on='gene')
df = pd.merge(df,stdf,on='gene')
print(len(ppidf),len(scdf),len(stdf),len(df))


# In[8]:


root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/disease/'
disease_source = 'OT_CAD'
diseasedf = pd.read_csv(root+disease_source+'.tsv',sep='\t')
pos_genes = diseasedf['symbol']
pos_genes = pos_genes.map(aliases2stringId)
pos_genes =  pos_genes.map(stringId2name)
print(len(diseasedf),len(pos_genes))
df['label'] = df['gene'].apply(lambda x: 1 if x in pos_genes.to_list() else -1)


# In[15]:


X = df[['ppi_1d', 'ppi_2d', 'ppi_3d', 'ppi_4d', 'ppi_re','sc_1d',
       'sc_2d', 'sc_3d', 'sc_4d', 'sc_re', 'st_1d', 'st_2d', 'st_3d', 'st_4d',
       'st_re']].to_numpy()
y = df['label'].to_numpy()

permut = np.random.permutation(len(y))
X = X[permut]
y = y[permut]


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out

model = Net(15,20,2)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train the neural network
epochs = 100
train_losses = []
test_losses = []
for epoch in range(epochs):
    # Training phase
    model.train()
    optimizer.zero_grad()
    inputs = torch.Tensor(X_train)
    labels = torch.Tensor(y_train).long()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation phase
    model.eval()
    inputs = torch.Tensor(X_test)
    labels = torch.Tensor(y_test).long()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    test_losses.append(loss.item())

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Plot the learning curve
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.save('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/classifier/NN/train curve.png')
plt.show()

# Evaluate the neural network on the test set
model.eval()
inputs = torch.Tensor(X_test)
labels = torch.Tensor(y_test).long()
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)
accuracy = (predicted == labels).sum().item() / len(labels)
print(f'Test Accuracy: {accuracy:.4f}')


# Draw the confusion matrix
cm = confusion_matrix(labels, predicted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.save('/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/classifier/NN/fusion.png')
plt.show()