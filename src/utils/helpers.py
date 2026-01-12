import os, sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
import torch.nn as nn



def plot_confusion_matrix(linear_probe_acc, labels: List= ['labels_1', 'labels_2', 'labels_s'], components: List= ['u_12', 'u_21', 's']):

    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    

    # Extract the arrays from the dictionary in the specified order
    arrays_to_stack = [linear_probe_acc[key] for key in components]

    # Stack the arrays vertically to create the 4x3 matrix
    result_matrix = np.vstack(arrays_to_stack)
    cm = result_matrix
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=components, ax=axes)
    axes.set_title('Linear Probe Accuracy for Components')
    axes.set_xlabel('Labels')
    axes.set_ylabel('Components')

    plt.tight_layout()
    plt.show()
    return fig


def plot_pairwise_confusion_matrices(linear_probe_acc, M, components: List= ['u_12', 'u_21', 's'], pairs: List= None):
    """
    Plot M*(M-1)/2 confusion matrices, one per modality pair.

    Rows: labels
    Columns: components
    Values: linear probe accuracy
    """

    # reconstruct component & label keys (same set)
    label_keys = components.copy()
    comp_keys = components.copy()

    # build full accuracy matrix (rows=labels, cols=components)
    A = np.stack([linear_probe_acc[k] for k in label_keys], axis=0)
    x_shape = M if M % 2 else M // 2
    y_shape = M - 1 if (M - 1) % 2 else M // 2
    
    x_shape, y_shape = (y_shape, x_shape) if x_shape > y_shape else (x_shape, y_shape)
    fig, axes = plt.subplots(x_shape, y_shape, figsize=(5 * y_shape, 6 * x_shape))
    
    for pair_id, (i, j) in enumerate(pairs):
        
        pair_name = f"{i+1} vs {j+1}"
        
        # columns: u_ij, u_ji, s_ij
        col_keys = [f"u_{i+1}{j+1}", f"u_{j+1}{i+1}", f"s_{i+1}{j+1}"]
        col_idx = [comp_keys.index(k) for k in col_keys]

        # rows: same pairwise labels
        row_keys = col_keys
        row_idx = [label_keys.index(k) for k in row_keys]

        submat = A[np.ix_(row_idx, col_idx)]


        axes_id_x = pair_id // y_shape
        axes_id_y = pair_id % y_shape

        ax = axes[axes_id_x, axes_id_y] if x_shape > 1 and y_shape > 1 else axes[max(axes_id_x, axes_id_y)]
        sns.heatmap(
            submat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=col_keys,
            yticklabels=row_keys,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Linear Probe – Pairwise Confusion ({pair_name})")
        ax.set_xlabel("Components")
        ax.set_ylabel("Labels")
        plt.tight_layout()
    plt.show()
    return fig

def linear_probe(train_data, train_labels, test_data, test_labels):
    # Train logistic regression
    clf = LogisticRegression(max_iter= 10000)
    clf.fit(train_data, train_labels)
    
    # Predict and compute accuracy
    labels_pred = clf.predict(test_data)
    acc = accuracy_score(test_labels, labels_pred) * 100  # Convert to percentage
    
    return acc

def non_linear_probe(train_data, train_labels, test_data, test_labels):

    input_dim = train_data.shape[1]
 
    output_dim = len(np.unique(train_labels))

    model = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.FloatTensor(train_data)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_data_tensor = torch.FloatTensor(test_data)
    test_labels_tensor = torch.LongTensor(test_labels)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data_tensor)
        loss = criterion(outputs, train_labels_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        total = test_labels_tensor.size(0)
        correct = (predicted == test_labels_tensor).sum().item()
        acc = (correct / total) * 100  # Convert to percentage

    return acc


# extract all the train-data and labels to have them ready for linear probing

def extract_latents_and_labels_2m(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data_m1, data_m2, labels_1, labels_2, labels_s) in enumerate(loader):
            dim_shape = data_m1.shape[-1] # dimension of original Z1 or Z2 
            data_m1 = data_m1.to(device)
            data_m2 = data_m2.to(device)
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_s = labels_s.to(device)
            
            outputs = model(data_m1, data_m2)
            u_12 = outputs['Z1'][0]
            s_21 = outputs['Z2'][1]
            u_21 = outputs['Z2'][0]
            s_12 = outputs['Z1'][1]
            
            if batch_idx == 0:
                # input data
                all_x12 = data_m1[:, 0, :dim_shape // 2]
                all_x21 = data_m2[:, 0, :dim_shape // 2]
                all_xs12 = data_m1[:, 0, dim_shape // 2:]
                all_xs21 = data_m2[:, 0, dim_shape // 2:]
                # latents
                all_u12 = u_12
                all_s21 = s_21
                all_u21 = u_21
                all_s12 = s_12
                all_labels_1 = labels_1
                all_labels_2 = labels_2
                all_labels_s = labels_s
            else:
                all_x12 = torch.cat([all_x12, data_m1[:, 0, :dim_shape // 2]], dim=0)
                all_x21 = torch.cat([all_x21, data_m2[:, 0, :dim_shape // 2]], dim=0)
                all_xs12 = torch.cat([all_xs12, data_m1[:, 0, dim_shape // 2:]], dim=0)
                all_xs21 = torch.cat([all_xs21, data_m2[:, 0, dim_shape // 2:]], dim=0)
                all_u12 = torch.cat([all_u12, u_12], dim=0)
                all_s21 = torch.cat([all_s21, s_21], dim=0)
                all_u21 = torch.cat([all_u21, u_21], dim=0)
                all_s12 = torch.cat([all_s12, s_12], dim=0)
                all_labels_1 = torch.cat([all_labels_1, labels_1], dim=0)
                all_labels_2 = torch.cat([all_labels_2, labels_2], dim=0)
                all_labels_s = torch.cat([all_labels_s, labels_s], dim=0)
    data_dict = {
        'x_12': all_x12.cpu().numpy(),
        'x_21': all_x21.cpu().numpy(),
        'xs_12': all_xs12.cpu().numpy(),
        'xs_21': all_xs21.cpu().numpy(),
        'u_12': all_u12.cpu().numpy(),
        's_21': all_s21.cpu().numpy(),
        'u_21': all_u21.cpu().numpy(),
        's_12': all_s12.cpu().numpy(),
        'labels_1': all_labels_1.cpu().numpy(),
        'labels_2': all_labels_2.cpu().numpy(),
        'labels_s': all_labels_s.cpu().numpy()}
    return data_dict

def extract_latents_and_labels(model, loader, device):
    # Initialize tensors to store all the latents & labels
    M = len(loader.dataset[0][0])  # number of modalities
    U_chunks = [[[] for _ in range(M)] for _ in range(M)]
    S_chunks = [[[] for _ in range(M)] for _ in range(M)]
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, labels_u, labels_s) in enumerate(loader):
            temp_b = X[0].shape[0]  # batch size
            dim_shape = X[0].shape[-1] # dimension of original Z1 or Z2 
            X = [X[m].to(device) for m in range(len(X))]
            
            outputs = model(X)
            
            for m1 in range(M):
                for m2 in range(M):
                    if m1 != m2:
                        U_chunks[m1][m2].append(outputs['U'][:, m1, m2, :].cpu().numpy())
                        S_chunks[m1][m2].append(outputs['S_view'][:, m1, m2, :].cpu().numpy())
            if batch_idx == 0:
                Labels_U = {k: v.detach().clone() for k, v in labels_u.items()}
                Labels_S = {k: v.detach().clone() for k, v in labels_s.items()}
            else:
                Labels_U = {k: torch.cat([Labels_U[k], labels_u[k]], dim=0) for k, v in labels_u.items()}
                Labels_S = {k: torch.cat([Labels_S[k], labels_s[k]], dim=0) for k, v in labels_s.items()}

    # U_final[m1][m2] will be (N, D) where N = total samples
    U_final = [[None for _ in range(M)] for _ in range(M)]
    S_final = [[None for _ in range(M)] for _ in range(M)]

    for m1 in range(M):
        for m2 in range(M):
            if m1 == m2:
                continue
            U_final[m1][m2] = np.concatenate(U_chunks[m1][m2], axis=0)
            S_final[m1][m2] = np.concatenate(S_chunks[m1][m2], axis=0)
    data_dict = {
        'U': U_final,
        'S': S_final,
        'Labels_U': {k: v.cpu().numpy() for k, v in Labels_U.items()},
        'Labels_S': {k: v.cpu().numpy() for k, v in Labels_S.items()}
    }
    return data_dict
