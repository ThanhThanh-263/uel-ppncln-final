import os
import pandas as pd

file1_path = r"../datadet/cleveland.data"
file2_path = r"../datadet/processed.cleveland.data"

# Đọc dữ liệu từ hai file
df1 = pd.read_csv(file1_path, header=None, encoding='latin1')
df2 = pd.read_csv(file2_path, header=None, encoding='latin1')

# Đảm bảo thư mục Datacsv tồn tại
os.makedirs("../Datacsv/dataset", exist_ok=True)

# Xuất ra Excel với hai sheet
excel_path = r"../Datacsv/dataset/heart_disease_data.xlsx"
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df1.to_excel(writer, sheet_name="Raw Data", index=False)
    df2.to_excel(writer, sheet_name="Processed Data", index=False)

# Trả về đường dẫn tệp Excel
print(f"File đã được lưu tại: {excel_path}")
