#!/usr/bin/env python3
"""
Analyze quantized features as individual 512-bit feature vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def read_feature_vectors(file_path="/tmp/physic_quantized_features.txt", vector_size=512):
    """
    Read quantized features as individual 512-bit vectors.
    
    Args:
        file_path: Path to the quantized features file
        vector_size: Size of each feature vector (default: 512)
        
    Returns:
        numpy array of shape (num_vectors, vector_size)
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        # Read raw bytes
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Convert to numpy array
        all_features = np.frombuffer(raw_data, dtype=np.uint8)
        
        # Check if data length is divisible by vector_size
        if len(all_features) % vector_size != 0:
            print(f"⚠️  Warning: Data length ({len(all_features)}) is not divisible by vector_size ({vector_size})")
            print(f"   Truncating to {len(all_features) // vector_size * vector_size} bytes")
            all_features = all_features[:len(all_features) // vector_size * vector_size]
        
        # Reshape into vectors
        num_vectors = len(all_features) // vector_size
        feature_vectors = all_features.reshape(num_vectors, vector_size)
        
        print(f"✓ Successfully read {len(all_features)} bytes from {file_path}")
        print(f"  Reshaped into {num_vectors} feature vectors of size {vector_size}")
        
        return feature_vectors
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def analyze_vector_distributions(feature_vectors):
    """
    Analyze the distribution of 0s and 1s in each vector.
    
    Args:
        feature_vectors: numpy array of shape (num_vectors, vector_size)
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        return
    
    print("\n" + "="*60)
    print("FEATURE VECTOR DISTRIBUTION ANALYSIS")
    print("="*60)
    
    num_vectors, vector_size = feature_vectors.shape
    print(f"Total vectors: {num_vectors}")
    print(f"Vector size: {vector_size} bits")
    
    # Analyze each vector's distribution
    vector_distributions = []
    for i, vector in enumerate(feature_vectors):
        zeros = np.sum(vector == 0)
        ones = np.sum(vector == 1)
        percentage_ones = (ones / vector_size) * 100
        vector_distributions.append(percentage_ones)
        
        if i < 5:  # Show first 5 vectors
            print(f"Vector {i+1:3d}: {zeros:3d} zeros, {ones:3d} ones ({percentage_ones:5.1f}% ones)")
    
    if num_vectors > 5:
        print(f"  ... and {num_vectors - 5} more vectors")
    
    # Overall statistics
    vector_distributions = np.array(vector_distributions)
    print(f"\nOverall Statistics:")
    print(f"  Mean percentage of 1s: {vector_distributions.mean():.2f}%")
    print(f"  Std deviation: {vector_distributions.std():.2f}%")
    print(f"  Min percentage: {vector_distributions.min():.2f}%")
    print(f"  Max percentage: {vector_distributions.max():.2f}%")
    
    return vector_distributions

def analyze_vector_differences(feature_vectors):
    """
    Analyze how different the feature vectors are from each other.
    
    Args:
        feature_vectors: numpy array of shape (num_vectors, vector_size)
    """
    if feature_vectors is None or len(feature_vectors) < 2:
        print("❌ Need at least 2 vectors to analyze differences")
        return
    
    print("\n" + "="*60)
    print("FEATURE VECTOR DIFFERENCE ANALYSIS")
    print("="*60)
    
    num_vectors, vector_size = feature_vectors.shape
    print(f"Analyzing differences between {num_vectors} vectors...")
    
    # Calculate Hamming distances between consecutive vectors
    hamming_distances = []
    for i in range(num_vectors - 1):
        # Count bit differences between consecutive vectors
        distance = np.sum(feature_vectors[i] != feature_vectors[i + 1])
        hamming_distances.append(distance)
    
    hamming_distances = np.array(hamming_distances)
    
    print(f"\nHamming Distances (consecutive vectors):")
    print(f"  Mean distance: {hamming_distances.mean():.1f} bits")
    print(f"  Std deviation: {hamming_distances.std():.1f} bits")
    print(f"  Min distance: {hamming_distances.min()} bits")
    print(f"  Max distance: {hamming_distances.max()} bits")
    print(f"  Expected random distance: {vector_size / 2} bits")
    
    # Calculate percentage differences
    percentage_differences = (hamming_distances / vector_size) * 100
    print(f"\nPercentage Differences:")
    print(f"  Mean: {percentage_differences.mean():.1f}%")
    print(f"  Std deviation: {percentage_differences.std():.1f}%")
    
    # Check if vectors are significantly different
    random_expected = 50.0  # 50% difference for random binary vectors
    mean_diff = percentage_differences.mean()
    
    if abs(mean_diff - random_expected) < 5:  # Within 5% of random
        print(f"  ✅ Vectors are significantly different (close to random)")
    else:
        print(f"  ⚠️  Vectors may be too similar (expected ~50% difference)")
    
    return hamming_distances, percentage_differences

def find_unique_vectors(feature_vectors):
    """
    Find how many unique vectors exist.
    
    Args:
        feature_vectors: numpy array of shape (num_vectors, vector_size)
    """
    if feature_vectors is None:
        return
    
    print("\n" + "="*60)
    print("UNIQUE VECTOR ANALYSIS")
    print("="*60)
    
    num_vectors, vector_size = feature_vectors.shape
    
    # Convert vectors to tuples for hashing (numpy arrays aren't hashable)
    vector_tuples = [tuple(vector) for vector in feature_vectors]
    
    # Count unique vectors
    unique_vectors = set(vector_tuples)
    num_unique = len(unique_vectors)
    
    print(f"Total vectors generated: {num_vectors}")
    print(f"Unique vectors: {num_unique}")
    print(f"Uniqueness ratio: {num_unique/num_vectors*100:.1f}%")
    
    if num_unique == num_vectors:
        print("  ✅ All vectors are unique!")
    elif num_unique > num_vectors * 0.9:
        print("  ✅ High uniqueness (>90%)")
    elif num_unique > num_vectors * 0.5:
        print("  ⚠️  Moderate uniqueness (50-90%)")
    else:
        print("  ❌ Low uniqueness (<50%) - vectors may be repeating")
    
    # Check for exact duplicates
    vector_counter = Counter(vector_tuples)
    duplicates = [vector for vector, count in vector_counter.items() if count > 1]
    
    if duplicates:
        print(f"\nDuplicate vectors found: {len(duplicates)}")
        for i, duplicate in enumerate(duplicates[:3]):  # Show first 3
            count = vector_counter[duplicate]
            print(f"  Duplicate {i+1}: appears {count} times")
    else:
        print(f"\n✅ No duplicate vectors found")
    
    return num_unique, len(duplicates)

def visualize_analysis(feature_vectors, vector_distributions, hamming_distances, save_plot=True):
    """
    Create visualizations of the analysis.
    
    Args:
        feature_vectors: numpy array of feature vectors
        vector_distributions: array of percentage of 1s in each vector
        hamming_distances: array of Hamming distances between consecutive vectors
        save_plot: whether to save the plot
    """
    if feature_vectors is None:
        return
    
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PHYSEC Feature Vector Analysis', fontsize=16)
    
    num_vectors, vector_size = feature_vectors.shape
    
    # Plot 1: First few vectors as binary patterns
    axes[0, 0].imshow(feature_vectors[:10], cmap='binary', aspect='auto')
    axes[0, 0].set_title('First 10 Feature Vectors (Binary Pattern)')
    axes[0, 0].set_xlabel('Bit Position')
    axes[0, 0].set_ylabel('Vector Index')
    
    # Plot 2: Distribution of 1s in each vector
    axes[0, 1].hist(vector_distributions, bins=20, alpha=0.7, color='blue')
    axes[0, 1].set_title('Distribution of 1s in Vectors')
    axes[0, 1].set_xlabel('Percentage of 1s (%)')
    axes[0, 1].set_ylabel('Number of Vectors')
    axes[0, 1].axvline(50, color='red', linestyle='--', alpha=0.7, label='50% (Random)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Hamming distances between consecutive vectors
    if hamming_distances is not None:
        axes[0, 2].plot(hamming_distances, 'g-', alpha=0.7)
        axes[0, 2].set_title('Hamming Distances (Consecutive Vectors)')
        axes[0, 2].set_xlabel('Vector Pair Index')
        axes[0, 2].set_ylabel('Hamming Distance (bits)')
        axes[0, 2].axhline(vector_size/2, color='red', linestyle='--', alpha=0.7, label=f'Random ({vector_size//2} bits)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Vector similarity matrix (first 20 vectors)
    if num_vectors >= 2:
        similarity_matrix = np.zeros((min(20, num_vectors), min(20, num_vectors)))
        for i in range(min(20, num_vectors)):
            for j in range(min(20, num_vectors)):
                if i != j:
                    # Calculate similarity as percentage of matching bits
                    similarity = np.sum(feature_vectors[i] == feature_vectors[j]) / vector_size * 100
                    similarity_matrix[i, j] = similarity
        
        im = axes[1, 0].imshow(similarity_matrix, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Vector Similarity Matrix (First 20)')
        axes[1, 0].set_xlabel('Vector Index')
        axes[1, 0].set_ylabel('Vector Index')
        plt.colorbar(im, ax=axes[1, 0], label='Similarity (%)')
    
    # Plot 5: Cumulative uniqueness over time
    if num_vectors > 1:
        cumulative_unique = []
        seen_vectors = set()
        for i in range(num_vectors):
            vector_tuple = tuple(feature_vectors[i])
            seen_vectors.add(vector_tuple)
            cumulative_unique.append(len(seen_vectors))
        
        axes[1, 1].plot(cumulative_unique, 'purple', linewidth=2)
        axes[1, 1].set_title('Cumulative Unique Vectors')
        axes[1, 1].set_xlabel('Vector Index')
        axes[1, 1].set_ylabel('Cumulative Unique Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Entropy over time (sliding window)
    if num_vectors > 10:
        window_size = min(10, num_vectors // 10)
        entropies = []
        for i in range(num_vectors - window_size + 1):
            window = feature_vectors[i:i+window_size]
            # Calculate entropy of the window
            total_ones = np.sum(window)
            total_bits = window.size
            p1 = total_ones / total_bits
            p0 = 1 - p1
            
            if p0 > 0 and p1 > 0:
                entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
                entropies.append(entropy)
            else:
                entropies.append(0)
        
        axes[1, 2].plot(entropies, 'orange', linewidth=2)
        axes[1, 2].set_title(f'Entropy Over Time (Window: {window_size})')
        axes[1, 2].set_xlabel('Window Start Index')
        axes[1, 2].set_ylabel('Entropy (bits)')
        axes[1, 2].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Perfect (1.0 bit)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_file = "/tmp/physic_feature_vectors_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Plot saved to: {plot_file}")
    
    plt.show()

def save_detailed_analysis(feature_vectors, vector_distributions, hamming_distances, output_file="/tmp/physic_feature_vectors_analysis.txt"):
    """
    Save detailed analysis to a text file.
    
    Args:
        feature_vectors: numpy array of feature vectors
        vector_distributions: array of percentage of 1s in each vector
        hamming_distances: array of Hamming distances
        output_file: output file path
    """
    if feature_vectors is None:
        return
    
    try:
        with open(output_file, 'w') as f:
            f.write("# PHYSEC Feature Vectors Analysis\n")
            f.write(f"# Generated from: /tmp/physic_quantized_features.txt\n")
            f.write(f"# Analysis date: {__import__('datetime').datetime.now()}\n")
            f.write("#\n")
            
            num_vectors, vector_size = feature_vectors.shape
            f.write(f"Total vectors: {num_vectors}\n")
            f.write(f"Vector size: {vector_size} bits\n")
            f.write(f"Total bits: {num_vectors * vector_size}\n\n")
            
            # Vector distributions
            f.write("VECTOR DISTRIBUTIONS:\n")
            f.write("=" * 50 + "\n")
            for i, pct in enumerate(vector_distributions):
                zeros = vector_size - int(pct * vector_size / 100)
                ones = int(pct * vector_size / 100)
                f.write(f"Vector {i+1:3d}: {zeros:3d} zeros, {ones:3d} ones ({pct:5.1f}% ones)\n")
            
            f.write(f"\nDistribution Statistics:\n")
            f.write(f"  Mean: {vector_distributions.mean():.2f}%\n")
            f.write(f"  Std Dev: {vector_distributions.std():.2f}%\n")
            f.write(f"  Min: {vector_distributions.min():.2f}%\n")
            f.write(f"  Max: {vector_distributions.max():.2f}%\n\n")
            
            # Hamming distances
            if hamming_distances is not None:
                f.write("HAMMING DISTANCES:\n")
                f.write("=" * 50 + "\n")
                for i, distance in enumerate(hamming_distances):
                    percentage = (distance / vector_size) * 100
                    f.write(f"Vectors {i+1}-{i+2}: {distance:3d} bits different ({percentage:5.1f}%)\n")
                
                f.write(f"\nDistance Statistics:\n")
                f.write(f"  Mean: {hamming_distances.mean():.1f} bits\n")
                f.write(f"  Std Dev: {hamming_distances.std():.1f} bits\n")
                f.write(f"  Min: {hamming_distances.min()} bits\n")
                f.write(f"  Max: {hamming_distances.max()} bits\n")
                f.write(f"  Expected random: {vector_size // 2} bits\n\n")
            
            # Sample vectors
            f.write("SAMPLE VECTORS:\n")
            f.write("=" * 50 + "\n")
            for i in range(min(5, num_vectors)):
                vector = feature_vectors[i]
                binary_str = ''.join(map(str, vector))
                hex_str = hex(int(binary_str, 2))[2:]
                f.write(f"Vector {i+1}:\n")
                f.write(f"  Binary: {binary_str[:100]}...\n")
                f.write(f"  Hex: {hex_str[:50]}...\n")
                f.write(f"  Distribution: {np.sum(vector==0)} zeros, {np.sum(vector==1)} ones\n\n")
        
        print(f"✓ Detailed analysis saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving detailed analysis: {e}")

def main():
    """Main function."""
    print("PHYSEC Feature Vector Analysis")
    print("=" * 60)
    
    # Read feature vectors
    feature_vectors = read_feature_vectors()
    
    if feature_vectors is None:
        return
    
    # Analyze distributions
    vector_distributions = analyze_vector_distributions(feature_vectors)
    
    # Analyze differences
    hamming_distances, percentage_differences = analyze_vector_differences(feature_vectors)
    
    # Find unique vectors
    num_unique, num_duplicates = find_unique_vectors(feature_vectors)
    
    # Create visualizations
    try:
        visualize_analysis(feature_vectors, vector_distributions, hamming_distances)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("  (This is okay - you can still analyze the data)")
    
    # Save detailed analysis
    save_detailed_analysis(feature_vectors, vector_distributions, hamming_distances)
    
    print("\n" + "="*60)
    print("✅ FEATURE VECTOR ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  📊 Total vectors: {feature_vectors.shape[0]}")
    print(f"  🔢 Vector size: {feature_vectors.shape[1]} bits")
    print(f"  ✨ Unique vectors: {num_unique}")
    print(f"  🔄 Duplicates: {num_duplicates}")
    print(f"  📈 Mean 1s percentage: {vector_distributions.mean():.1f}%")
    print(f"  📏 Mean Hamming distance: {hamming_distances.mean():.1f} bits")
    
    print(f"\nFiles created:")
    print(f"  📄 Raw binary: /tmp/physic_quantized_features.txt")
    print(f"  📊 Analysis plot: /tmp/physic_feature_vectors_analysis.png")
    print(f"  📝 Detailed analysis: /tmp/physic_feature_vectors_analysis.txt")

if __name__ == "__main__":
    main()