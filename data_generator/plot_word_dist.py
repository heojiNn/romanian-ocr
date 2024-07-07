import matplotlib.pyplot as plt

def plot_word_count_distribution(file_path):
    """
    Reads a .txt file and plots the word count distribution.
    
    Args:
        file_path (str): The path to the .txt file.
    """
    word_counts = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line by tab and get the text part
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text = parts[1]
                # Count the number of words in the text
                word_count = len(text.split())
                word_counts.append(word_count)

    # Calculate frequencies and percentages
    max_count = max(word_counts)
    count_frequencies = {i: word_counts.count(i) for i in range(1, max_count + 1)}
    total_count = len(word_counts)
    count_percentages = {k: (v / total_count) * 100 for k, v in count_frequencies.items()}

    # Print the percentages
    print("Word Count Distribution (relative percentages):")
    for k, v in count_percentages.items():
        print(f"{k} words: {v:.2f}%")

    # Plot the word count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=range(1, max_count + 2), edgecolor='black', align='left')
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max_count + 1))
    plt.grid(axis='y')
    plt.show()

# Usage example
plot_word_count_distribution('benchmarks/balcesu_benchmark/balcesu_fragment_test/labels.txt')
