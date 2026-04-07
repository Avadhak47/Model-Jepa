import numpy as np

# --- Action Constants ---
ACTION_TRANSLATE = 0
ACTION_ROTATE = 1
ACTION_SCALE = 2
ACTION_FILL_COLOR = 3
ACTION_CONNECT = 4
ACTION_BOUNDING_BOX = 5
# Note: we can expand this list easily later

# --- A. Spatial Manipulation ---
def translate(grid: np.ndarray, slot_mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Moves an isolated object in the given (dy, dx) direction."""
    new_grid = grid.copy()
    if not np.any(slot_mask):
        return new_grid
        
    obj_pixels = grid[slot_mask]
    y_idx, x_idx = np.where(slot_mask)
    new_y = y_idx + dy
    new_x = x_idx + dx
    
    # Option A: Hard collisions (Fail silently if out of bounds)
    h, w = grid.shape
    if np.any(new_y < 0) or np.any(new_y >= h) or np.any(new_x < 0) or np.any(new_x >= w):
        return grid # Silent failure, no change
        
    new_grid[slot_mask] = 0 # Clear old position
    new_grid[new_y, new_x] = obj_pixels # Write new position
    return new_grid

def rotate_object(grid: np.ndarray, slot_mask: np.ndarray, k: int=1) -> np.ndarray:
    """Rotates an object by k*90 degrees around its bounding box center."""
    new_grid = grid.copy()
    if not np.any(slot_mask): 
        return new_grid
    
    y_idx, x_idx = np.where(slot_mask)
    min_y, max_y = y_idx.min(), y_idx.max()
    min_x, max_x = x_idx.min(), x_idx.max()
    
    patch = grid[min_y:max_y+1, min_x:max_x+1]
    patch_mask = slot_mask[min_y:max_y+1, min_x:max_x+1]
    
    obj_patch = np.where(patch_mask, patch, 0)
    rotated_patch = np.rot90(obj_patch, k=k)
    rotated_mask = np.rot90(patch_mask, k=k)
    
    # Find new placement (centered at original center)
    cy, cx = (min_y + max_y) / 2.0, (min_x + max_x) / 2.0
    rh, rw = rotated_patch.shape
    
    new_min_y = int(round(cy - rh / 2.0))
    new_min_x = int(round(cx - rw / 2.0))
    
    # Option A: Check bounds
    if new_min_y < 0 or new_min_y + rh > grid.shape[0] or new_min_x < 0 or new_min_x + rw > grid.shape[1]:
        return grid
        
    new_grid[slot_mask] = 0 # Clear old
    dest_slice = new_grid[new_min_y:new_min_y+rh, new_min_x:new_min_x+rw]
    # Only overwrite where the rotated mask is True to prevent overwriting with 0s
    dest_slice[rotated_mask] = rotated_patch[rotated_mask]
    
    return new_grid

# --- B. Color & Painting ---
def fill_color(grid: np.ndarray, slot_mask: np.ndarray, color: int) -> np.ndarray:
    """Paints an entire object a solid color."""
    new_grid = grid.copy()
    new_grid[slot_mask] = color
    return new_grid

# --- C. Topology & Logic ---
def connect(grid: np.ndarray, slot_mask_a: np.ndarray, slot_mask_b: np.ndarray, color: int) -> np.ndarray:
    """Draws a line connecting the centroids of A and B using the specified color."""
    new_grid = grid.copy()
    if not np.any(slot_mask_a) or not np.any(slot_mask_b): 
        return new_grid
    
    ya, xa = np.where(slot_mask_a)
    ca_y, ca_x = int(np.mean(ya)), int(np.mean(xa))
    
    yb, xb = np.where(slot_mask_b)
    cb_y, cb_x = int(np.mean(yb)), int(np.mean(xb))
    
    # Simple line drawing using interpolation
    dy = cb_y - ca_y
    dx = cb_x - ca_x
    steps = max(abs(dy), abs(dx))
    
    if steps == 0:
        new_grid[ca_y, ca_x] = color
        return new_grid
        
    y_inc = dy / steps
    x_inc = dx / steps
    
    y, x = ca_y, ca_x
    for _ in range(int(steps) + 1):
        # Clip to grid just to be perfectly safe
        ry, rx = int(round(y)), int(round(x))
        if 0 <= ry < grid.shape[0] and 0 <= rx < grid.shape[1]:
            new_grid[ry, rx] = color
        y += y_inc
        x += x_inc
        
    return new_grid

def bounding_box(grid: np.ndarray, slot_mask: np.ndarray, color: int) -> np.ndarray:
    """Draws a perimeter around the outermost edges of a slot."""
    new_grid = grid.copy()
    if not np.any(slot_mask):
        return new_grid
        
    y_idx, x_idx = np.where(slot_mask)
    min_y, max_y = y_idx.min(), y_idx.max()
    min_x, max_x = x_idx.min(), x_idx.max()
    
    # Top and bottom edges
    new_grid[min_y, min_x:max_x+1] = color
    new_grid[max_y, min_x:max_x+1] = color
    # Left and right edges
    new_grid[min_y:max_y+1, min_x] = color
    new_grid[min_y:max_y+1, max_x] = color
    
    return new_grid

# --- Testing Script ---
def test_dsl():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:4, 2:4] = 1 # Red square
    mask_a = (grid == 1)
    
    grid[7:9, 7:9] = 2 # Blue square
    mask_b = (grid == 2)
    
    print("Original State:")
    print(grid)
    
    print("\n1. Test Translate (A Down by 2)")
    print(translate(grid, mask_a, 2, 0))
    
    print("\n2. Test Translate Hard Collision (A Up by 5 -> Silent Fail)")
    print(translate(grid, mask_a, -5, 0))
    
    print("\n3. Test Fill Color (Paint B Green=3)")
    print(fill_color(grid, mask_b, 3))
    
    print("\n4. Test Connect (Draw line from A to B in Yellow=4)")
    print(connect(grid, mask_a, mask_b, 4))
    
    print("\n5. Test Bounding Box (Box around B in Orange=5)")
    print(bounding_box(grid, mask_b, 5))

if __name__ == '__main__':
    test_dsl()
