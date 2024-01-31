#!/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/hbdm_env2/bin/python3.8
#SBATCH -J ppi_scendo_st_4
#SBATCH -o ppi_scendo_st_4.log
#SBATCH -e ppi_scendo_st_4.err
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80G


import torch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

dataset = 'ppi_scendo_st_4'
######## define label
root= '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/data/'
df = pd.read_csv(root+dataset+'.csv')
disease_root = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/thesis/HBDM/data/disease/'
disease_source = 'DIS_CAD'
diseasedf = pd.read_csv(disease_root+disease_source+'.tsv',sep='\t')
pos_genes = diseasedf['Gene']
df['label'] = df['gene'].apply(lambda x: 1 if x in pos_genes.to_list() else 0)

# disease_source = 'OT_CAD'
# diseasedf = pd.read_csv(disease_root+disease_source+'.tsv',sep='\t')
# pos_genes = diseasedf['symbol']
# df['label'] = df['gene'].apply(lambda x: 1 if x in pos_genes.to_list() else 0)

# disease_source = 'cad_literature'
# diseasedf = pd.read_csv(disease_root+disease_source+'.csv')
# pos_genes = diseasedf['gene']
# df['label'] = df['gene'].apply(lambda x: 1 if x in pos_genes.to_list() else 0)

# genticpath = '/novo/omdb/pds02/PDS2843/data/sprint_tid_ascvd/gzn/310_targets/CAD-Genetic-Target-Discovery-2023_010323.xlsx'
# geneticdf=pd.read_excel(genticpath,sheet_name='MASTER')
# pos_genes = geneticdf['Gene']
# df['label'] = df['gene'].apply(lambda x: 1 if x in pos_genes.to_list() else 0)

print(len(df[df['label']==1]))
avg1 = []
avg2 = []
avg3 = []
avg4 = []
for i in range(5):
    ######## get X and y
    y = df['label'].to_numpy()
    filtered_df = df.filter(regex='^(ppi|sc|st)')
    X = filtered_df.to_numpy()

    # Generate a random permutation of the indices
    perm = np.random.permutation(len(X))
    # Use the permutation to shuffle X and y
    X = X[perm]
    y = y[perm]

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    print('train: ',X_train.shape,y_train.shape)
    print('train pos: ',len(np.where(y_train == 1.)[0]))
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]

    # randomly select a subset of the 0 samples
    idx_0_balanced = np.random.choice(idx_0, size=len(idx_1), replace=False)

    # concatenate the indices of the balanced dataset
    idx_balanced = np.concatenate((idx_0_balanced, idx_1))

    # create the balanced dataset
    X_train = X_train[idx_balanced]
    y_train = y_train[idx_balanced]

    print('banlance train: ',X_train.shape,y_train.shape)
    print('banlance train pos: ',len(np.where(y_train == 1.)[0]))
    print('test: ',X_test.shape,y_test.shape)
    print('test pos: ',len(np.where(y_test == 1.)[0]))
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # number of features (len of X cols)
    input_dim = X.shape[1]
    # number of hidden layers
    hidden_layers = 25

    output_dim = 1


    clf = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_layers),
        nn.Linear(in_features=hidden_layers, out_features=10),
        nn.Linear(in_features=10, out_features=5),
        nn.Linear(in_features=5, out_features=output_dim),
        nn.ReLU()
    ).to(device)

    #
    loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

    # Create an optimizer
    optimizer = torch.optim.SGD(params=clf.parameters(), 
                                lr=0.1)

    # Set the number of epochs
    epochs = 1000

    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Calculate accuracy (a classification metric)
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc

    best_f1 = 0.0
    not_improved_count = 0
    for epoch in range(epochs):
        ### Training
        clf.train()

        # 1. Forward pass (model outputs raw logits)
        y_logits = clf(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
    
        # 2. Calculate loss/accuracy
        loss = loss_fn(torch.sigmoid(y_logits), y_train) 
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred) 
        f1 = f1_score(y_train.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        clf.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = clf(X_test).squeeze() 
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
            test_f1 = f1_score(y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy())

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | F1 score: {f1:.2f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%, Test F1 score: {test_f1:.2f}")
        
        # Stop training if F1 score doesn't improve
        if test_f1 > best_f1:
            best_f1 = test_f1
            not_improved_count = 0
        else:
            not_improved_count += 1
            if not_improved_count == 20:
                print(f"F1 score hasn't improved in {not_improved_count} epochs, stopping training.")
                break

    ##### metrics
    y_pre = torch.round(torch.sigmoid(clf(X_test).squeeze()))
    print('ann f1: ',f1_score(y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy()))
    avg1.append(f1_score(y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy()))
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    svc.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    y_pred = svc.predict(X_test.detach().cpu().numpy())
    print('svc f1: ',f1_score(y_test.detach().cpu().numpy(), y_pred))
    avg2.append(f1_score(y_test.detach().cpu().numpy(), y_pred))
    rf = RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                bootstrap=True,
                n_jobs=1,
            )
    rf.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    y_pred = rf.predict(X_test.detach().cpu().numpy())
    print('rf f1: ',f1_score(y_test.detach().cpu().numpy(), y_pred))
    avg3.append(f1_score(y_test.detach().cpu().numpy(), y_pred))
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    y_pred = knn.predict(X_test.detach().cpu().numpy())
    print('knn f1: ',f1_score(y_test.detach().cpu().numpy(), y_pred))
    avg4.append(f1_score(y_test.detach().cpu().numpy(), y_pred))

print(np.mean(avg1))
print(np.mean(avg2))
print(np.mean(avg3))
print(np.mean(avg4))