import pandas as pd
import os

# Using relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Assuming notebook is in scripts/
RAW_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'Ad_table_extra.csv')
IMAGE_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw', 'confirmed_fronts')
PROCESSED_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'cleaned_dvm_data.csv')

df = pd.read_csv(RAW_CSV_PATH)
df.columns = df.columns.str.strip()

# Clean missing and extreme values
df = df.dropna(subset=['Price', 'Maker', 'Reg_year', 'Runned_Miles', 'Color', 'Genmodel'])
Q1, Q3 = df['Price'].quantile(0.25), df['Price'].quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[(df['Price'] >= (Q1 - 1.5 * IQR)) & (df['Price'] <= (Q3 + 1.5 * IQR))]
df_cleaned = df_cleaned[df_cleaned['Price'] > 500]

print("Scanning nested image directories...")
image_records = []

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith('.jpg'):
            # Parse: Bentley$$Arnage$$2003$$Black$$10_1$$8$$image_28.jpg
            parts = file.split('$$')

            # The Adv_ID in the CSV format is '10_1$$8' (parts[4] + '$$' + parts[5])
            if len(parts) >= 6:
                adv_id = f"{parts[4]}$${parts[5]}"
                rel_path = os.path.relpath(os.path.join(root, file), IMAGE_DIR)

                image_records.append({
                    'Adv_ID': adv_id,
                    'Image_Path': rel_path
                })

df_images = pd.DataFrame(image_records)
df_images = df_images.drop_duplicates(subset=['Adv_ID'])  # Strictly 1-to-1 mapping

# Merge STRICTLY on Adv_ID
df_final = pd.merge(df_cleaned, df_images, on='Adv_ID', how='inner')

df_final.to_csv(PROCESSED_CSV_PATH, index=False)
print(f"Clean data saved to {PROCESSED_CSV_PATH} with shape: {df_final.shape}")