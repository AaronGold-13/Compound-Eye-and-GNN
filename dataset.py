import os.path as osp
import os
from tqdm import tqdm
import torch, numpy as np, math
from torch_geometric.data import Dataset, download_url, Data
import pandas as pd

class CompoundEyeDataset_CE37(Dataset):
    def __init__(self, root, filename='BigDataset_CE_37.csv', transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.filename = filename
        super(CompoundEyeDataset_CE37, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0], header=None).reset_index()

        return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        for index, point in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Get node features
            point_ = point.values.tolist()

            node_feats = self._get_node_features(point_)
            edge_weights = self._get_edge_weights(point_, self.get_coefficients_dict())
            edge_index = self._get_adjacency_info(point_)
            label = self._get_labels(point_)

            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_weights,
                        y=label)
            torch.save(data,
                       os.path.join(self.processed_dir,
                       f'data_{index}.pt'))

    def get_coefficients_dict(self):
        rows = [
            4,  # row 1
            5,  # row 2
            6,  # row 3
            7,  # row 4
            6,  # row 5
            5,  # row 6
            4  # row 7
        ]

        def generate_coordinates(rows):
            dist_se = 1.0
            coordinates = []
            offset_x = 0
            for row_idx, num_nodes in enumerate(rows):
                for i in range(num_nodes):
                    if row_idx == 0:
                        offset_x = 0
                        offset_y = 0
                    elif row_idx == 1:
                        offset_x = -dist_se/2
                    elif row_idx == 2:
                        offset_x = -dist_se
                    elif row_idx == 3:
                        offset_x = -1.5 * dist_se
                    elif row_idx == 4:
                        offset_x = -dist_se
                    elif row_idx == 5:
                        offset_x = -dist_se/2
                    elif row_idx == 6:
                        offset_x = 0
                    offset_y = row_idx * np.sqrt(1 - 0.25)
                    x = i + offset_x
                    coordinates.append((x, offset_y))
            return coordinates

        def euclidean_distance(coord1, coord2):
            return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

        coordinates = generate_coordinates(rows)

        distances = np.zeros((len(coordinates), len(coordinates)))

        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = euclidean_distance(coordinates[i], coordinates[j])
                distances[i][j] = dist
                distances[j][i] = dist

        max_distance = np.max(distances)
        normalized_distances = distances / max_distance

        out_dict = {}
        for i in range(37):
            for j in range(37):
                if i < j:
                    out_dict[f'{i}and{j}'] = normalized_distances[i][j]
        print(out_dict)
        return out_dict
      
    def _get_node_features(self, coordinates):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for i in range(0, len(coordinates)-3, 2):
            if coordinates[i: i + 2] == [0, 0]:
                continue
            else:
                feature = coordinates[i: i + 2]
                all_node_feats.append(feature)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_adjacency_info(self, coordinates):
        edge_indices_new = []
        node_indices = []
        for i in range(0, len(coordinates) - 3, 2):
            if coordinates[i: i + 2] == [0, 0]:
                continue
            else:
                node_indices.append(i // 2)
        node_indices_new = [i for i in range(len(node_indices))]
        for index_1 in node_indices_new:
            for index_2 in node_indices_new:
                if index_2 > index_1:
                    edge_indices_new.append([index_1, index_2])
        edge_indices_new = torch.tensor(edge_indices_new)
        edge_indices_new = edge_indices_new.t().to(torch.long).view(2, -1)

        return edge_indices_new

    def _get_edge_weights(self, coordinates, coefficients):

        node_indices = []
        edge_indices = []
        for i in range(0, len(coordinates) - 3, 2):
            if coordinates[i: i + 2] == [0, 0]:
                continue
            else:
                node_indices.append(i // 2)
        for index_1 in node_indices:
            for index_2 in node_indices:
                if index_2 > index_1:
                    edge_indices.append([index_1, index_2])
        edge_weights = [0 for i in range(len(edge_indices))]
        for i, pair in enumerate(edge_indices):
            edge_weights[i] = coefficients[f'{pair[0]}and{pair[1]}']
        edge_weights = torch.tensor(edge_weights)

        return edge_weights

    def _get_labels(self, coordinates):
        label = np.asarray(coordinates[74:])
        return torch.tensor(label, dtype=torch.float)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir,
                                       f'data_{idx}.pt'))
        return data
