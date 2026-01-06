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

def extract_latents_and_labels(model, loader, device):
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
