import os
import json
import torch
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from torch.utils.data import Dataset

class ARCGrid:
    """Represents a single ARC 2D grid and provides methods to analyze its topology."""
    
    def __init__(self, grid_array: list, task_id: str = ""):
        self.grid = np.array(grid_array)
        self.height, self.width = self.grid.shape
        self.task_id = task_id
        self.background_color = self._detect_background()

    def _detect_background(self):
        """Dynamically detect background color (most frequent color)."""
        counts = np.bincount(self.grid.flatten(), minlength=10)
        return int(np.argmax(counts))

    def _crop_from_mask(self, mask, color, hypothesis):
        """Helper to safely extract the bounding box of a mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        crop = self.grid[rmin:rmax+1, cmin:cmax+1].copy()
        crop_mask = mask[rmin:rmax+1, cmin:cmax+1]
        crop[~crop_mask] = self.background_color
        
        return {
            "crop": crop,
            "color": color,
            "bbox": (int(rmin), int(rmax), int(cmin), int(cmax)),
            "shape": crop.shape,
            "metadata": {
                "hypothesis": hypothesis,
                "task_id": self.task_id,
                "bg_color": self.background_color
            }
        }

    def extract_atomic_primitives(self):
        """Hypothesis 1: Contiguous regions of the same color (8-connectivity)."""
        extracted = []
        for color in range(10):
            if color == self.background_color:
                continue
            mask = (self.grid == color)
            if not np.any(mask):
                continue
                
            labeled_mask, num_features = ndimage.label(mask, structure=np.ones((3, 3)))
            for i in range(1, num_features + 1):
                obj = self._crop_from_mask(labeled_mask == i, color, "Atomic (Single-Color)")
                if obj: extracted.append(obj)
        return extracted

    def extract_composite_objects(self):
        """Hypothesis 2: Multi-color foreground clusters."""
        extracted = []
        mask = (self.grid != self.background_color)
        if not np.any(mask):
            return extracted
            
        labeled_mask, num_features = ndimage.label(mask, structure=np.ones((3, 3)))
        for i in range(1, num_features + 1):
            comp_mask = (labeled_mask == i)
            unique_colors = np.unique(self.grid[comp_mask])
            # Only keep if it contains multiple foreground colors (single color is caught by Atomic)
            if len(unique_colors) > 1:
                obj = self._crop_from_mask(comp_mask, -1, "Composite (Multi-Color)") # -1 denotes mixed color
                if obj: extracted.append(obj)
        return extracted
        
    def extract_grid_partitions(self):
        """Hypothesis 3: Sub-grids partitioned by spanning lines."""
        extracted = []
        if self.height < 5 or self.width < 5: 
            return extracted
            
        # Find horizontal lines
        h_lines = []
        for r in range(self.height):
            if len(np.unique(self.grid[r, :])) == 1 and self.grid[r, 0] != self.background_color:
                h_lines.append(r)
                
        # Find vertical lines
        v_lines = []
        for c in range(self.width):
            if len(np.unique(self.grid[:, c])) == 1 and self.grid[0, c] != self.background_color:
                v_lines.append(c)
                
        if len(h_lines) > 0 or len(v_lines) > 0:
            # Reconstruct the partition mask and invert to find sub-grids
            line_mask = np.zeros_like(self.grid, dtype=bool)
            for r in h_lines: line_mask[r, :] = True
            for c in v_lines: line_mask[:, c] = True
            
            subgrid_mask = ~line_mask & (self.grid != self.background_color)
            labeled_mask, num_features = ndimage.label(subgrid_mask, structure=np.ones((3, 3)))
            for i in range(1, num_features + 1):
                obj = self._crop_from_mask(labeled_mask == i, -1, "Partitioned Sub-Grid")
                if obj: extracted.append(obj)
                
        return extracted

    def extract_all_hypotheses(self):
        return self.extract_atomic_primitives() + self.extract_composite_objects() + self.extract_grid_partitions()

class ObjectNormalizer:
    """Handles the padding of variable-sized objects to a fixed tensor size to preserve pristine geometry."""
    
    def __init__(self, target_size: int = 15):
        self.target_size = target_size

    def normalize(self, obj_dict: dict) -> dict:
        crop = obj_dict["crop"]
        h, w = crop.shape
        bg_color = obj_dict["metadata"]["bg_color"]
        
        h = min(h, self.target_size)
        w = min(w, self.target_size)
        
        padded_crop = np.full((self.target_size, self.target_size), bg_color, dtype=np.int32)
        valid_mask = np.zeros((self.target_size, self.target_size), dtype=bool)
        
        r_start = (self.target_size - h) // 2
        c_start = (self.target_size - w) // 2
        
        padded_crop[r_start:r_start+h, c_start:c_start+w] = crop[:h, :w]
        valid_mask[r_start:r_start+h, c_start:c_start+w] = (crop[:h, :w] != bg_color)
        
        return {
            "tensor": padded_crop,
            "mask": valid_mask,
            "color": obj_dict["color"],
            "metadata": obj_dict["metadata"]
        }

class ObjectExtractor:
    """Orchestrates the scanning of datasets and extraction of all fundamental primitives."""
    
    def __init__(self, data_paths: list, normalizer: ObjectNormalizer):
        self.data_paths = data_paths
        self.normalizer = normalizer
        self.primitive_library = []
        
    def process_all(self):
        total_files = 0
        for path in self.data_paths:
            if not os.path.exists(path):
                continue
                
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            for file in tqdm(files, desc=f"Extracting from {os.path.basename(path)}"):
                with open(os.path.join(path, file), 'r') as f:
                    task = json.load(f)
                    self._process_task(task, task_id=file.split('.')[0])
                total_files += 1
                
        print(f"Extraction complete. Found {len(self.primitive_library)} total hypotheses.")
        self._deduplicate()
        
    def _process_task(self, task: dict, task_id: str):
        all_pairs = task.get('train', []) + task.get('test', [])
        for idx, pair in enumerate(all_pairs):
            for grid_key in ['input', 'output']:
                if grid_key in pair:
                    grid_obj = ARCGrid(pair[grid_key], task_id=f"{task_id}_{idx}_{grid_key}")
                    objects = grid_obj.extract_all_hypotheses()
                    for obj in objects:
                        normalized_obj = self.normalizer.normalize(obj)
                        self.primitive_library.append(normalized_obj)

    def _deduplicate(self):
        unique_objects = []
        seen_hashes = set()
        
        for obj in self.primitive_library:
            shape_hash = hash(obj["mask"].tobytes())
            if shape_hash not in seen_hashes:
                seen_hashes.add(shape_hash)
                unique_objects.append(obj)
                
        print(f"Deduplication complete. Kept {len(unique_objects)} unique objects.")
        self.primitive_library = unique_objects
        
    def save_library(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tensors = torch.from_numpy(np.stack([obj["tensor"] for obj in self.primitive_library])).long()
        masks = torch.from_numpy(np.stack([obj["mask"] for obj in self.primitive_library])).bool()
        colors = torch.tensor([obj["color"] for obj in self.primitive_library]).long()
        metadata = [obj["metadata"] for obj in self.primitive_library]
        
        payload = {
            "tensors": tensors,
            "masks": masks,
            "colors": colors,
            "metadata": metadata
        }
        torch.save(payload, output_path)
        print(f"Saved multi-hypothesis vocabulary to {output_path}")

class PrimitiveDataset(Dataset):
    def __init__(self, library_path: str):
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"Library not found at {library_path}")
            
        payload = torch.load(library_path)
        self.tensors = payload["tensors"].float().unsqueeze(1)
        self.masks = payload["masks"]
        self.colors = payload["colors"]
        
    def __len__(self): return len(self.tensors)
    def __getitem__(self, idx):
        return {"state": self.tensors[idx], "valid_mask": self.masks[idx], "color_id": self.colors[idx]}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_paths = [
        os.path.join(base_dir, 'ARC-AGI', 'data', 'training'),
        os.path.join(base_dir, 'arc_data', 're-arc', 'arc_original', 'training'),
        os.path.join(base_dir, 'arc_data', 're-arc', 'arc_original', 'evaluation')
    ]
    
    valid_paths = [p for p in data_paths if os.path.exists(p)]
    if valid_paths:
        normalizer = ObjectNormalizer(target_size=15)
        extractor = ObjectExtractor(data_paths=valid_paths, normalizer=normalizer)
        extractor.process_all()
        extractor.save_library(os.path.join(base_dir, 'arc_data', 'primitive_library.pt'))
