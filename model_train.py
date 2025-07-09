import torch, torch_geometric, numpy as np, math, random
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch, Data
from torch_geometric.nn import GATConv, TransformerConv, GraphConv
from dataset import CompoundEyeDataset_CE37
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib.pyplot as plt


class GraphNormalizer:
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def fit(self, dataset):
        all_nodes = []
        all_outputs = []

        for data in dataset:
            nodes = data.x.numpy()
            output = data.y.numpy()
            mask = ~((nodes[:, 0] == 0) & (nodes[:, 1] == 0))
            filtered_nodes = nodes[mask]
            all_nodes.append(filtered_nodes)
            all_outputs.append([output])
        all_nodes = np.concatenate(all_nodes, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        self.node_scaler.fit(all_nodes)
        self.output_scaler.fit(all_outputs)

    def transform(self, data):
        nodes, output = data.x, data.y
        mask = ~((nodes[:, 0] == 0) & (nodes[:, 1] == 0))
        filtered_nodes = nodes[mask]
        normalized_nodes = self.node_scaler.transform(filtered_nodes.numpy())
        normalized_output = self.output_scaler.transform(output.numpy().reshape(1, -1))
        final_normalized_nodes = nodes.clone()
        final_normalized_nodes[mask] = torch.Tensor(normalized_nodes)

        normalized_data = Data(x=final_normalized_nodes,
                               edge_index=data.edge_index,
                               edge_attr=data.edge_attr,
                               y=torch.tensor(normalized_output.flatten(),
                                              dtype=torch.float32))

        return normalized_data

    def inverse_transform_output(self, normalized_output):
        return self.output_scaler.inverse_transform(normalized_output)


# model
class GNN_Model(nn.Module):
    def __init__(self, in_channels, base_channels, out_channels):
        super(GNN_Model, self).__init__()
        self.base_channels = base_channels
        self.activation = nn.ELU(inplace=True)
        self.conv1 = GraphConv(in_channels, base_channels)
        self.conv2 = GATConv(base_channels, base_channels * 2, heads=4, concat=True, edge_dim=1)
        self.conv3 = GATConv(base_channels * 8, base_channels * 4, heads=4, concat=False, edge_dim=1)
        self.conv_trans = TransformerConv(4 * base_channels, 4 * base_channels, heads=4,
                                          concat=True, edge_dim=1)
        self.conv4 = GATConv(base_channels * 16, base_channels * 2, heads=4, concat=True, edge_dim=1)
        self.fc1 = nn.Linear(base_channels * 8, base_channels*2)
        self.fc_out = nn.Linear(base_channels*2, out_channels)

    def forward(self, batch):
        x, edge_index, edge_weights = batch.x.float(), batch.edge_index, batch.edge_attr.float()
        edge_attr = edge_weights.unsqueeze(-1)
        batch_indices = batch.batch
        x = self.activation(self.conv1(x, edge_index, edge_weights))
        x = self.activation(self.conv2(x, edge_index, edge_attr))
        x = self.activation(self.conv3(x, edge_index, edge_attr))
        x = self.activation(self.conv_trans(x, edge_index, edge_attr))
        x = self.activation(self.conv4(x, edge_index, edge_attr))
        
        x = torch_geometric.nn.global_add_pool(x, batch_indices)
        x = self.activation(self.fc1(x))
        out = self.fc_out(x)

        return out


def train(batches):
    model.train()
    total_loss = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).to(device)
        if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
            print("Output contains NaN or Inf")
            return
        loss = criterion(out,
                         torch.stack([batch[i].y for i in range(len(batch))]))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = total_loss / (len(batches))
    print(f'Training Loss: {train_loss}')
    sys.stdout.flush()

    return train_loss


def test(batches):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            out = model(batch).to(device)
            if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
                print("Output contains NaN or Inf")
                return
            loss = criterion(out,
                             torch.stack([batch[i].y for i in range(len(batch))]))
            total_loss += loss.item()

    return total_loss / (len(batches))


def eval_loss(real_vec, pred_vec):
    return pred_vec - real_vec


def dist_error(real_points, pred_points):

    count = 0
    error = 0
    rel_error = 0
    random.seed(13)
    pairs = [(random.randint(0, 109), random.randint(0, 109)) for _ in range(200)]
    real_distances = []
    pred_distances = []
    for pair in pairs:
        index_1 = pair[0]
        index_2 = pair[1]
        if index_1 == index_2:
            continue
        pred_point_1 = pred_points[index_1]
        pred_point_2 = pred_points[index_2]

        real_point_1 = real_points[index_1].squeeze(0)
        real_point_2 = real_points[index_2].squeeze(0)

        count += 1
        real_distance = math.dist(real_point_1, real_point_2)
        pred_distance = math.dist(pred_point_1, pred_point_2)
        real_distances.append(real_distance)
        pred_distances.append(pred_distance)

        diff = np.abs(real_distance-pred_distance)
        relat_diff = (diff / real_distance) * 100
        error += diff
        rel_error += relat_diff
        
    print("Distance measurement average absolute error: ", error / count, " mm")
    print("Distance measurement average relative error: ", rel_error / count, " %")
    sys.stdout.flush()
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10

    plt.figure(figsize=(6, 5))
    plt.ylabel('Distance, mm')
    plt.grid()
    plt.title('Visual Measurement')
    plt.plot(real_distances[:100], label="real values", marker='o', color='black')
    plt.plot(pred_distances[:100], label="predicted values", marker='^', color='red')
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_real_error(test_data, test_data_normalized):

    model.eval()
    final_loader = DataLoader(test_data_normalized, batch_size=1, shuffle=False)
    final_loader_denorm = DataLoader(test_data, batch_size=1, shuffle=False)
    final_loss = []
    real_output = []
    pred_output = []
    for i, (norm_batch, denorm_batch) in enumerate(zip(final_loader, final_loader_denorm)):
        print(f'For test point {i}: ')
        norm_batch = norm_batch.to(device)
        denorm_batch = denorm_batch.to(device)
        real_y = denorm_batch[0].y.detach().cpu()
        final_out = model(norm_batch).detach().cpu()
        final_out = torch.Tensor(normalizer.inverse_transform_output(final_out)).view(3).cpu()
        real_output.append(real_y)
        pred_output.append(final_out)
        print('Real coordinates: ', real_y)
        print('Predicted coordinates: ', final_out)
        final_loss.append(np.abs(eval_loss(final_out, real_y)))
        print('Loss: ', eval_loss(final_out, real_y))
    loss_x = 0
    loss_y = 0
    loss_z = 0
    for loss_ in final_loss:
        loss_x += loss_[0]
        loss_y += loss_[1]
        loss_z += loss_[2]
    print('Mean loss by coordinate x: ', loss_x / len(final_loss))
    print('Mean loss by coordinate y: ', loss_y / len(final_loss))
    print('Mean loss by coordinate z: ', loss_z / len(final_loss))
    print('Mean loss: ', (loss_x + loss_y + loss_z) / (3 * len(final_loss)))
    
    return real_output, pred_output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for training: {device}")

    # dataset
    dataset = CompoundEyeDataset_CE37(root='./data_CE37')
    dataset_dim = len(dataset)

    # dataset dividing
    torch_geometric.seed_everything(13)
    indices = torch.randperm(dataset_dim)
    train_size = int(0.955 * dataset_dim)
    validation_size = int(0.99 * dataset_dim)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:validation_size]
    valid_indices = indices[validation_size:]
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    valid_dataset = [dataset[i] for i in valid_indices]

    # normalization
    train_dataset_normalized = []
    test_dataset_normalized = []
    valid_dataset_normalized = []
    normalizer = GraphNormalizer()
    normalizer.fit(train_dataset)
    for data in train_dataset:
        normalized_data = normalizer.transform(data)
        train_dataset_normalized.append(normalized_data)
    for data in test_dataset:
        normalized_data = normalizer.transform(data)
        test_dataset_normalized.append(normalized_data)
    for data in valid_dataset:
        normalized_data = normalizer.transform(data)
        valid_dataset_normalized.append(normalized_data)

    batch_size = 128
    train_loader = DataLoader(train_dataset_normalized, batch_size=batch_size, shuffle=True)
    print(f'Number of train batches: {len(train_loader)}')
    test_loader = DataLoader(valid_dataset_normalized, batch_size=batch_size, shuffle=False)
    print(f'Number of test batches: {len(test_loader)}')

    model = GNN_Model(in_channels=2, base_channels=8, out_channels=3).to(device)
    model.load_state_dict(torch.load('model_epoch1499.pth', map_location=torch.device('cpu')))
    print('Model parameters number: ', sum(p.numel() for p in model.parameters()))

    criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0).to(device)
    weight_decay = 1e-4
    lr = 0.01
    gamma = 0.9975
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    epochs = 1500
    start_epoch = 0
    save_model_freq = 50
    check_real_error_freq = 100
    
    # train the model
    best_test_error = 0
    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch}/{epochs}')
        train_loss = train(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print("Learning Rate: ", current_lr)
        test_loss = test(test_loader)
        scheduler.step()
        print(f'Test Loss: {test_loss}')
        if epoch % check_real_error_freq == 0:
            real_points, pred_points = find_real_error()
        sys.stdout.flush()
    
    # check on test data
    real_points, pred_points = find_real_error(test_dataset, test_dataset_normalized)
    dist_error(real_points, pred_points)
