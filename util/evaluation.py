import os
import numpy as np
from difflib import SequenceMatcher

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cer(gt, pred):
    return levenshtein_distance(gt, pred) / len(gt)

def wer(gt, pred):
    gt_words = gt.split()
    pred_words = pred.split()
    return levenshtein_distance(gt_words, pred_words) / len(gt_words)

def normalized_edit_distance(gt, pred):
    return levenshtein_distance(gt, pred) / max(len(gt), len(pred))

def match_score(gt, pred):
    return 1 if gt == pred else 0

import re
def clean_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing spaces

def calculate_metrics(original_text, predicted_text):

    predicted_text = clean_whitespaces(predicted_text) # this should be done after inference...

    levenshtein_dist = levenshtein_distance(original_text, predicted_text)
    correct_characters = len(original_text) - levenshtein_dist
    total_characters = len(original_text)
    character_accuracy = correct_characters / total_characters

    ground_truth_words = original_text.split()
    predicted_words = predicted_text.split()

    total_words = len(ground_truth_words)
    word_accuracy = (total_words - levenshtein_distance(ground_truth_words, predicted_words)) / total_words

    max_len = max(len(original_text), len(predicted_text))
    normalized_levenshtein_distance = levenshtein_dist / max_len
    levenshtein_accuracy = 1 - normalized_levenshtein_distance

    recognition_time = np.nan  # Placeholder, since we don't have timing information

    return {
        'Levenshtein Distance': levenshtein_dist,
        'Character Accuracy': character_accuracy,
        'Word Accuracy': word_accuracy,
        'Normalized Levenshtein Distance': normalized_levenshtein_distance,
        'Levenshtein Accuracy': levenshtein_accuracy,
        'Levenshtein Percentage': normalized_levenshtein_distance * 100,
        'CER': cer(original_text, predicted_text),
        'WER': wer(original_text, predicted_text),
        'Recognition Time': recognition_time,
    }

def process_files(directory, ground_truth_file):
    # Read the ground truth text
    with open(ground_truth_file, 'r', encoding='utf-8') as gt_file:
        original_text = gt_file.read()

    results = []

    # Iterate over all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename != os.path.basename(ground_truth_file):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                predicted_text = file.read()

            metrics = calculate_metrics(original_text, predicted_text)
            results.append({
                'Method': os.path.splitext(filename)[0],
                'Levenshtein Distance': metrics['Levenshtein Distance'],
                'Character Accuracy': metrics['Character Accuracy'],
                'Word Accuracy': metrics['Word Accuracy'],
                'Normalized Levenshtein Distance': metrics['Normalized Levenshtein Distance'],
                'Levenshtein Accuracy': metrics['Levenshtein Accuracy'],
                'CER': metrics['CER'],
                'WER': metrics['WER'],
            })

    # Sort results by Word Accuracy, then by Levenshtein Distance
    results.sort(key=lambda x: (-x['Character Accuracy'], x['Levenshtein Distance'], -x['Word Accuracy']))

    # Format results into the specified output string
    output = "## Evaluation Table:\n"
    output += "| Method         | LDist    | Char Acc | Word Acc | Norm LD  | Lev Acc  | CER    | WER   |\n"
    output += "|----------------|----------|----------|----------|----------|----------|--------|-------|\n"
    for result in results:
        output += (f"| {result['Method']:<14} | {result['Levenshtein Distance']:<8} | "
                   f"{result['Character Accuracy']:<8.2f} | {result['Word Accuracy']:<8.2f} | "
                   f"{result['Normalized Levenshtein Distance']:<8.4f} | {result['Levenshtein Accuracy']:<8.4f} | "
                   f"{result['CER']:<6.4f} | {result['WER']:<6.4f} |\n")

    return output

# Iterate over all prediction text files:
directory = 'benchmarks/balcesu_benchmark'
# Have one (biased!) groundtruth as benchmark:
ground_truth_file = 'benchmarks/balcesu_benchmark/gt.txt'

# Process files and generate results
result_string = process_files(directory, ground_truth_file)
print(result_string)
