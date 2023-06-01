import torch
import numpy as np

class SingleReasoningPath:
    def __init__(self, root_node, topk):
        self.kpath = list()
        self.root_node = root_node
        for _ in range(topk):
            self.kpath.append([('ROOT_NODE', root_node, 1.)])
        self.topk = topk

    def add_new_step(self, rel, node, prob, k):
        self.kpath[k].append((rel, node, prob))

    def get_current_nodes(self):
        nodes = list()
        for kp in self.kpath:
            nodes.append((kp[-1][1], kp[-1][2]))
        return nodes

    def get_root_node(self):
        return self.root_node

    def get_reasoning_path(self):
        return self.kpath




class AllReasoningPath:
    def __init__(self):
        self.all_path = dict()

    def set_root_nodes(self, root_nodes, topk):
        self.all_path = dict()
        for node in root_nodes:
            path = SingleReasoningPath(node, topk)
            self.all_path[node] = path

    def get_current_nodes(self, root_node=None):
        if root_node:
            return self.all_path[root_node].get_current_nodes()
        else:
            return {k: v.get_current_nodes() for k, v in self.all_path.items()}

    def add_new_step(self, root_node, k, rel, node, prob):
        self.all_path[root_node].add_new_step(rel, node, prob, k)

    def get_all_reasoning_path(self):
        return {k: v.get_reasoning_path() for k,v in self.all_path.items()}


# Define a function named find_triplets. It will be used inside the Custom layer
def find_triplets(list_of_triplets, start=None, rel=None, end=None):
  # Initialize an empty list to store the matching triplets
  result = []
  # Loop through each triplet in the input list
  for triplet in list_of_triplets:
    # Check if the triplet matches the start, rel, end parameters
    # If any parameter is None, it means any value is acceptable
    if (start is None or triplet[0] == start) and (rel is None or triplet[1] == rel) and (end is None or triplet[2] == end):
      # Add the matching triplet to the result list
      result.append(triplet)
  # Return the result list
  return result

def extract_all_relations_for_a_node(node_name, triplets):
  return np.unique([r for _, r, _ in find_triplets(triplets, start=node_name)]).tolist()

def extract_values_from_tensor(tensor, indices):
    """
    Extracts values from a tensor based on the given indices.

    Arguments:
    tensor -- A torch.Tensor of shape (M, N) containing real numbers.
    indices -- A list of lists specifying the indices for each row of the tensor.

    Returns:
    result -- A list of lists of tensors containing values from the tensor
              corresponding to the given indices.
    """

    result = []

    # Iterate over the rows of the tensor and corresponding indices
    for row_indices in indices:
        row_result = []

        # Extract values from the row based on the given indices
        for index in row_indices:
            row_result.append(tensor[index])

        result.append(torch.stack(row_result))

    return result