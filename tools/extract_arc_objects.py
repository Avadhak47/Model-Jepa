import os
import json
import torch
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from torch.utils.data import Dataset

class ARCGrid:
    def __init__(self, grid_array: list, task_id: str = ""):
        self.grid = np.array(grid_array)
        self.height, self.width = self.grid.shape
        self.task_id = task_id
        self.background_color = self._detect_background()

    def _detect_background(self):
        counts = np.bincount(self.grid.flatten(), minlength=10)
        return int(np.argmax(counts))

    def _extract_normalized_object(self, mask, color, hypothesis):
        """
        New Logic:
        1. Crop to bounding box.
        2. Remap background inside crop to 0.
        3. Align to Top-Left (0,0).
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 1. Precise Crop
        crop_mask = mask[rmin:rmax+1, cmin:cmax+1]
        raw_crop = self.grid[rmin:rmax+1, cmin:cmax+1].copy()
        
        # 2. Force Background to Black (0)
        # Only pixels inside the mask keep their color. Everything else is 0.
        clean_crop = np.zeros_like(raw_crop)
        clean_crop[crop_mask] = raw_crop[crop_mask]
        
        # 3. Handle specific color remapping (for multi-hypothesis)
        if color != -1:
            # For single-color atoms, we can normalize the color to 1 
            # so NMF only learns the SHAPE. We will keep true color for now though.
            pass

        return {
            "crop": clean_crop,
            "mask": crop_mask,
            "color": color,
            "metadata": {
                "hypothesis": hypothesis,
                "task_id": self.task_id
            }
        }

    def extract_all_hypotheses(self):
        extracted = []
        # Single Color Islands
        for color in range(10):
            if color == self.background_color: continue
            mask = (self.grid == color)
            if not np.any(mask): continue
            labeled, num = ndimage.label(mask, structure=np.ones((3, 3)))
            for i in range(1, num + 1):
                obj = self._extract_normalized_object(labeled == i, color, "Single-Color")
                if obj: extracted.append(obj)
        
        # Multi-Color Islands
        fg_mask = (self.grid != self.background_color)
        if np.any(fg_mask):
            labeled, num = ndimage.label(fg_mask, structure=np.ones((3, 3)))
            for i in range(1, num + 1):
                obj = self._extract_normalized_object(labeled == i, -1, "Composite")
                if obj: extracted.append(obj)
        return extracted

class ObjectNormalizer:
    def __init__(self, target_size: int = 15):
        self.target_size = target_size

    def normalize(self, obj_dict: dict) -> dict:
        crop = obj_dict["crop"]
        h, w = crop.shape
        
        # Align to TOP-LEFT (0,0)
        h_fit = min(h, self.target_size)
        w_fit = min(w, self.target_size)
        
        padded = np.zeros((self.target_size, self.target_size), dtype=np.int32)
        padded[0:h_fit, 0:w_fit] = crop[0:h_fit, 0:w_fit]
        
        mask = (padded > 0)
        
        return {
            "tensor": padded,
            "mask": mask,
            "color": obj_dict["color"],
            "metadata": obj_dict["metadata"]
        }

class ObjectExtractor:
    def __init__(self, data_paths: list, normalizer: ObjectNormalizer):
        self.data_paths = data_paths
        self.normalizer = normalizer
        self.primitive_library = []
        
    def process_all(self):
        for path in self.data_paths:
            if not os.path.exists(path): continue
            files = [f for f in os.listdir(path) if f.endswith('.json')]
            for file in tqdm(files, desc=f"Scanning {os.path.basename(path)}"):
                try:
                    with open(os.path.join(path, file), 'r') as f:
                        task = json.load(f)
                        self._process_task(task, file.split('.')[0])
                except: continue
        self._deduplicate()
        
    def _process_task(self, task: dict, task_id: str):
        all_pairs = task.get('train', []) + task.get('test', [])
        for idx, pair in enumerate(all_pairs):
            for k in ['input', 'output']:
                if k in pair:
                    grid = ARCGrid(pair[k], task_id=f"{task_id}_{idx}_{k}")
                    for obj in grid.extract_all_hypotheses():
                        self.primitive_library.append(self.normalizer.normalize(obj))

    def _deduplicate(self):
        unique = []
        seen = set()
        for obj in self.primitive_library:
            # Hash the actual pixel values [15,15]
            h = hash(obj["tensor"].tobytes())
            if h not in seen:
                seen.add(h)
                unique.append(obj)
        print(f"Deduplicated: {len(self.primitive_library)} -> {len(unique)} unique patterns.")
        self.primitive_library = unique

    def save_library(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "tensors": torch.from_numpy(np.stack([o["tensor"] for o in self.primitive_library])).long(),
            "masks": torch.from_numpy(np.stack([o["mask"] for o in self.primitive_library])).bool(),
            "colors": torch.tensor([o["color"] for o in self.primitive_library]).long(),
            "metadata": [o["metadata"] for o in self.primitive_library]
        }
        torch.save(payload, output_path)

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [
        os.path.join(base, 'ARC-AGI', 'data', 'training'),
        os.path.join(base, 'arc_data', 're-arc', 'arc_original', 'training'),
        os.path.join(base, 'arc_data', 'arc-heavy', 'data') # Added ARC-Heavy
    ]
    valid = [p for p in paths if os.path.exists(p)]
    normalizer = ObjectNormalizer(target_size=15)
    extractor = ObjectExtractor(data_paths=valid, normalizer=normalizer)
    extractor.process_all()
    extractor.save_library(os.path.join(base, 'arc_data', 'primitive_library.pt'))
