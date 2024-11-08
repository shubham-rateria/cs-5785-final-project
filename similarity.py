from skimage.metrics import structural_similarity
import numpy as np
import os
import pickle
import time

def calculate_similarity(args):
    i, j, images = args
    if i == j:
        return i, j, 1
    else:
        score = 1 - structural_similarity(images[i], images[j], full=True)[0]
        return i, j, score
    
def calculate_similarity_row(args):
    """Calculates similarity scores for a single row in the similarity matrix."""
    checkpoint_dir='./checkpoints'
    i, images = args
    
    checkpoint_file = os.path.join(checkpoint_dir, f'similarity_matrix_row{i}.pkl')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            row = pickle.load(f)
            return i, row
    
    
    start_time = time.time()
    print("row", i)
    num_images = len(images)
    row = np.zeros(num_images)
    
    for j in range(i, num_images):
        if i == j:
            row[j] = 1  # Diagonal element (similarity with itself)
        else:
            score = 1 - structural_similarity(images[i], images[j], full=True)[0]
            row[j] = score
            row[j] = row[j] if j == i else row[j]
    checkpoint_file = os.path.join(checkpoint_dir, f'similarity_matrix_row{i}.pkl')
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(row, f)
    end_time = time.time()
    
    print("time taken", (end_time - start_time)/1000)
    
    return i, row