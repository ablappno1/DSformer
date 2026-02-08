import torch

from torch_geometric.data import Data
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import math
from dataloaders.common import MultimodalDatasetMP, RegressionDatasetMP
from dataloaders.common import generate_site_species_vector
from .common import CellFormat

# --- Angle calculation auxiliary function ---
def compute_local_interactions(structure, radius=3.0):
    """
    Calculates sparse edges and angular features within a local area.

    Returns edge_index and edge_attr conforming to the PyG standard.
    """
    # Get all nearest neighbors (including periodic images)
    # get_all_neighbors Return: [ [ (neighbor, dist, index, image), ... ], ... ]
    all_neighbors = structure.get_all_neighbors(r=radius, include_index=True)
    
    edge_src_list = []
    edge_dst_list = []
    edge_angles_list = []
    
    # Traverse each central atom i
    for i, neighbors in enumerate(all_neighbors):
        # Skip if there are no neighbors.
        if len(neighbors) == 0:
            continue
            
        center_coords = structure.cart_coords[i]
        
        # Preprocessing: Extract the vectors and indices of all neighbors of the central atom.
        # n[2] is the global index of the neighboring atoms.
        n_vectors = [] 
        n_indices = []
        
        for n in neighbors:
            # Calculate the vector pointing from center i to neighbor j.
            vec = n[0].coords - center_coords
            n_vectors.append(vec)
            n_indices.append(n[2])
            
        # Traverse every edge of the central atom (i -> j)
        for idx_j, (vec_j, atom_j_idx) in enumerate(zip(n_vectors, n_indices)):
            
            # Record the index of this edge (Source, Destination)
            edge_src_list.append(i)
            edge_dst_list.append(atom_j_idx)
            
            # Calculate all bond angles (j-i-k) involved by this edge (i-j).
            angles_for_this_edge = []
            
            norm_j = np.linalg.norm(vec_j)
            
            for idx_k, vec_k in enumerate(n_vectors):
                if idx_j == idx_k: continue # Skip itself
                
                norm_k = np.linalg.norm(vec_k)
                
                if norm_j > 1e-6 and norm_k > 1e-6:
                    # cos_theta = (v_j . v_k) / (|v_j| * |v_k|)
                    cosine = np.dot(vec_j, vec_k) / (norm_j * norm_k)
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angle = math.acos(cosine) # radian system
                    angles_for_this_edge.append(angle)
            
            # Aggregation strategy
            # Here, the average value is taken as the "angular feature" of this side.
            # If the edge is isolated (does not form an angle), fill in 0 or pi/2.
            if angles_for_this_edge:
                avg_angle = np.mean(angles_for_this_edge)
            else:
                avg_angle = 0.0
                
            edge_angles_list.append(avg_angle)

    # Transform tensor
    if len(edge_src_list) > 0:
        edge_index = torch.tensor([edge_src_list, edge_dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_angles_list, dtype=torch.float).unsqueeze(-1) # [E, 1]
    else:
        # Handling the case where there are no edges (extremely rare).
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    return edge_index, edge_attr

# ---  make_data ---
def make_data(material, ATOM_NUM_UPPER, cell_format: CellFormat):
    if "final_structure" in material:
        structure = material['final_structure']
    elif "structure" in material:
        structure = material['structure']
    else:
        raise AttributeError("Material has no structure!")
        
    if cell_format == CellFormat.CONVENTIONAL:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    elif cell_format == CellFormat.PRIMITIVE:
        structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()

    if "material_id" in material:
        id = material['material_id']
    elif "file_id" in material:
        id = material['file_id']
    else:
        id = material['id']

    # --- Basic features ---
    atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    atom_fea = generate_site_species_vector(structure, ATOM_NUM_UPPER)
    
    data = Data(x=atom_fea, y=None, pos=atom_pos)
    
    # trans_vec must be augmented with the dimension [1, 3, 3] to match the batch logic.
    data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)
    
    data.sizes = torch.tensor([len(structure)], dtype=torch.long)
    data.material_id = id
    
    # --- Calculate local sparse angle information ---
    # radius=3.0 Å 
    local_edge_index, local_angles = compute_local_interactions(structure, radius=3.0)
    
    # Store the sparse graph in Data
    data.local_edge_index = local_edge_index  # [2, E]
    data.local_angles = local_angles          # [E, 1] (An average angle value for each side)

    if "target" in material:
        y = material['target']
        data.y = torch.tensor([y], dtype=torch.float)

    return data

# def make_data(material, ATOM_NUM_UPPER, cell_format:CellFormat):
#     if "final_structure" in material:
#         structure = material['final_structure']
#     elif "structure" in material:
#         structure = material['structure']
#     else:
#         raise AttributeError("Material has no structure!")
#     if cell_format == CellFormat.CONVENTIONAL:
#         structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
#     elif cell_format == CellFormat.PRIMITIVE:
#         structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
#         # # assert len(structure.cart_coords) == len(primitive.cart_coords), f"{len(structure.cart_coords)}, {len(primitive.cart_coords)}"


#     if "material_id" in material:
#         id = material['material_id']
#     elif "file_id" in material:
#         id = material['file_id']
#     else:
#         id = material['id']

#     atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
#     atom_fea = generate_site_species_vector(structure, ATOM_NUM_UPPER)
#     data = Data(x=atom_fea, y=None, pos=atom_pos)
#     data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float)[None]
#     data.material_id = id
#     data.sizes = torch.tensor([atom_pos.shape[0]], dtype=torch.long)
#     return data

class MultimodalDatasetMP_Latticeformer(MultimodalDatasetMP):
    def __init__(self, params, target_split, target_set=None, post_filter=None):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super(MultimodalDatasetMP_Latticeformer, self).__init__(target_split, target_set, post_filter)
    
    @property
    def processed_file_names(self):
        if self.use_primitive:
            return 'processed_data_latticeformer.pt'
        else:
            return 'processed_data_convcell_latticeformer.pt'

    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()


class RegressionDatasetMP_Latticeformer(RegressionDatasetMP):
    def __init__(self, target_split, target_set=None, cell_format:CellFormat=CellFormat.PRIMITIVE, post_filter=None):
        self.model_name = "latticeformer"
        super(RegressionDatasetMP_Latticeformer, self).__init__(target_split, target_set, cell_format, post_filter)
    
    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.cell_format)
    
    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()
