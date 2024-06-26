def normalize_and_save_text(file_path, output_file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the entire contents of the file
            text = file.read()
            # Remove line feeds and replace them with spaces
            normalized_text = text.replace('\n', ' ')
            
            # Save the normalized text to a new file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(normalized_text)
            
            print("Normalized text saved to:", output_file_path)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

# Provide the path to your text file
file_path = 'ocr/balcesu_benchmark/adobe.txt'

# Call the function to normalize text and save it to a new file
normalize_and_save_text(file_path, file_path)
