import requests
import os
import time
from urllib.parse import urljoin, urlparse
import re

# Base URL for Google Fonts API
api_url = "https://www.googleapis.com/webfonts/v1/webfonts"
api_key = "AIzaSyA3WUQ8TMmKMtpS5fTScpJqitcBSyI_aYY"  # Replace with your Google Fonts API key

# Output directory for fonts
output_dir = "data_generator/assets/fonts/google"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Checkpoint file to store the last downloaded font
checkpoint_file = "data_generator/assets/font_download_checkpoint.txt"

# Get the list of all fonts
params = {
    'key': api_key,
    'sort': 'alpha'
}
response = requests.get(api_url, params=params)
fonts_data = response.json()

# Get the total number of fonts
total_fonts = len(fonts_data['items'])
print(f"Total number of fonts: {total_fonts}")

# Function to download a font file
def download_font(url, font_name, style):
    try:
        font_file_name = f"{font_name}-{style}.ttf"
        font_file_path = os.path.join(output_dir, font_file_name)
        if not os.path.exists(font_file_path):
            font_response = requests.get(url)
            if font_response.status_code == 200:
                with open(font_file_path, "wb") as font_file:
                    font_file.write(font_response.content)
                print(f"Downloaded {font_file_name}")
                # Update the checkpoint file
                with open(checkpoint_file, "w") as checkpoint:
                    checkpoint.write(f"{font_name}-{style}")
            else:
                print(f"Failed to download {font_name} ({style}): {font_response.status_code}")
        else:
            print(f"Already exists: {font_file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {font_name} ({style}): {e}")

# Read the last downloaded font from the checkpoint file
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as checkpoint:
        last_downloaded = checkpoint.read().strip()
        print('last downloaded:', last_downloaded)
else:
    last_downloaded = ""

# Flag to determine whether to start downloading
start_downloading = False if last_downloaded else True

# Download all fonts
for font in fonts_data['items']:
    font_family = font['family']
    css_url = f"https://fonts.googleapis.com/css2?family={font_family.replace(' ', '+')}&display=swap"
    
    try:
        css_response = requests.get(css_url)
        if css_response.status_code == 200:
            css_content = css_response.text
            font_urls = re.findall(r'url\((https://fonts.gstatic.com/s/[^)]+\.ttf)\)', css_content)

            for url in font_urls:
                style = url.split('/')[-1].split('.')[0]
                font_name_style = f"{font_family}-{style}"

                # Check if we have reached the last downloaded font
                if font_name_style == last_downloaded:
                    start_downloading = True

                if start_downloading:
                    download_font(url, font_family, style)
                    time.sleep(1)  # Be polite to the server and avoid rate limiting
                else:
                    print(f"Skipping {font_name_style}, already downloaded.")
        else:
            print(f"Failed to fetch CSS for {font_family}: {css_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch CSS for {font_family}: {e}")

print("All fonts downloaded.")

