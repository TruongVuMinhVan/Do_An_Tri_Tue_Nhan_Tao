import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime as dt
import plotly.express as px
from IPython.display import HTML
from statsmodels.tsa.arima.model import ARIMA
from prettytable import PrettyTable  

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Tệp tại {file_path} không được tìm thấy.")
        return None

data = load_data(r"D:\STUDY\TTNT\ĐỒ ÁN\economic_data.csv")
if data is None:
    exit()

print ("-----*DATASET CỦA PHÂN TÍCH THẤT NGHIỆP TỪ 1991-2022*-----")
print (data)

data.info()
data.describe()

# Kiểm tra giá trị thiếu
print("Số lượng giá trị thiếu cho từng cột:")
print(data.isnull().sum())

# Chuyển đổi kiểu dữ liệu
data['Date'] = pd.to_datetime(data['Date'])

# Loại bỏ các hàng có giá trị thiếu NaN (Not a Number)
data.dropna(inplace=True)

# Vẽ biểu đồ đường tỷ lệ thất nghiệp theo thời gian
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Unemployment Rate (%)', data=data)
plt.title('Tỷ lệ Thất nghiệp Theo Thời gian (Biểu Đồ Đường)')
plt.xlabel('Ngày')
plt.ylabel('Tỷ lệ Thất nghiệp')
plt.show()

# Vẽ biểu đồ thanh cột
data['Year'] = data['Date'].dt.year  # Trích xuất năm(Year) từ ngày(Date)
plt.figure(figsize=(10, 6)) # (Chiều rộng, chiều cao)
sns.barplot(x='Year', y='Unemployment Rate (%)', data=data)
plt.title('Tỷ lệ Thất nghiệp Theo Năm (Biểu Đồ Cột)')
plt.xlabel('Năm')
plt.ylabel('Tỷ lệ Thất nghiệp')
plt.xticks(ticks=range(0, len(data['Year'].unique()), 5), labels=data['Year'].unique()[::5])
plt.show()

# Tính số lượng thất nghiệp
def calculate_unemployed(data):
    data['Unemployed'] = (data['Unemployment Rate (%)'] / 100) * data['Population']
    return data

# Kiểm tra kiểu dữ liêu côt Population
if data['Population'].dtype == 'object':
    data['Population'] = data['Population'].str.replace('B', '').astype(float) * 1e9 # (1e9 = 1x10^9)

data = calculate_unemployed(data)

# Đặt 'Date' làm chỉ số
data.set_index('Date', inplace=True)
data = data.asfreq('YE')

# Áp dụng mô hình ARIMA #(AutoRegressive Integrated Moving Average)
model = ARIMA(data['Unemployment Rate (%)'], order=(5, 1, 0))
model_fit = model.fit()

# Dự báo 12 tháng tiếp theo
forecast = model_fit.forecast(steps=12)

# Đặt ngày bắt đầu từ tháng 12/2022
start_date = pd.to_datetime('2022-12-01')
forecast_dates = pd.date_range(start=start_date, periods=12, freq='ME')

# Tạo DataFrame cho dự báo
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Unemployment Rate (%)': forecast})

# Hiển thị kết quả dự báo
print("\n-----------*DỰ BÁO TỶ LỆ THẤT NGHIỆP TỪ THÁNG 12/2022 ĐẾN THÁNG 11/2023*-----------")
print(forecast_df)

# Vẽ biểu đồ
plt.figure(figsize=(13, 6))

# Vẽ dữ liệu thực tế từ tháng 12/2022 trở đi
sns.lineplot(x=data.index[data.index >= start_date], y=data['Unemployment Rate (%)'][data.index >= start_date], label='Thực tế', color='red')

# Vẽ dữ liệu dự báo
sns.lineplot(x=forecast_df['Date'], y=forecast_df['Forecasted Unemployment Rate (%)'], label='Dự báo', color='blue', marker='o')

# Cài đặt tiêu đề và nhãn
plt.title('Dự báo Tỷ lệ Thất nghiệp từ tháng 12/2022 đến tháng 11/2023')
plt.xlabel('Ngày')
plt.ylabel('Tỷ lệ Thất nghiệp')
plt.axvline(x=start_date - pd.DateOffset(months=1), color='gray', linestyle='--', label='Điểm bắt đầu dự báo')  # Vẽ đường thẳng để chỉ điểm bắt đầu dự báo
plt.xlim([start_date - pd.DateOffset(months=1), forecast_df['Date'].iloc[-1]])  # Giới hạn trục x từ tháng 11/2022 đến tháng 11/2023
plt.legend()
plt.grid()
plt.show()



print("\n-----------***-----------")
# Kiểm tra ngoại lệ
print(data['Inflation Rate (%)'].describe())
sns.boxplot(x=data['Inflation Rate (%)'])
plt.title('Phân phối Tỷ lệ Lạm phát')
plt.show()

sns.boxplot(x=data['Unemployment Rate (%)'])
plt.title('Phân phối Tỷ lệ Thất nghiệp')
plt.show()

# **Thêm vào đây: BÁO CÁO TÓM TẮT SỬ DỤNG PrettyTable**
summary = data.describe()
summary_table = PrettyTable()
summary_table.field_names = ["Biến", "Số lượng", "Trung bình", "Độ lệch chuẩn", "Giá trị nhỏ nhất", 
                              "Phân vị thứ 25", "Trung vị", "Phân vị thứ 75", "Giá trị lớn nhất"]

# Thêm vào bảng báo cáo
for column in summary.columns:
    summary_table.add_row([column, 
                            int(summary[column]['count']),
                            f"{summary[column]['mean']:.2f}",
                            f"{summary[column]['std']:.2f}",
                            f"{summary[column]['min']:.2f}",
                            f"{summary[column]['25%']:.2f}",
                            f"{summary[column]['50%']:.2f}",
                            f"{summary[column]['75%']:.2f}",
                            f"{summary[column]['max']:.2f}"])

# In bảng tóm tắt
print("\n------------*BÁO CÁO TÓM TẮT*------------")
print(summary_table)

# Giải thích báo cáo tóm tắt
print("\n------------*GIẢI THÍCH BÁO CÁO TÓM TẮT*------------")

# Tỷ lệ thất nghiệp
print(f"Tỷ lệ thất nghiệp(Unemployment Rate (%)): ")
print(f"- Số lượng: {int(summary['Unemployment Rate (%)']['count'])} quan sát")
print(f"- Trung bình: {summary['Unemployment Rate (%)']['mean']:.2f}%")
print(f"- Độ lệch chuẩn: {summary['Unemployment Rate (%)']['std']:.2f}")
print(f"- Giá trị nhỏ nhất: {summary['Unemployment Rate (%)']['min']:.2f}%")
print(f"- Phân vị thứ 25: {summary['Unemployment Rate (%)']['25%']:.2f}%")
print(f"- Trung vị: {summary['Unemployment Rate (%)']['50%']:.2f}%")
print(f"- Phân vị thứ 75: {summary['Unemployment Rate (%)']['75%']:.2f}%")
print(f"- Giá trị lớn nhất: {summary['Unemployment Rate (%)']['max']:.2f}%")

# Dân số
print("\nDân số(Population): ")
print(f"- Số lượng: {int(summary['Population']['count'])} quan sát")
print(f"- Trung bình: {int(summary['Population']['mean']):,}")  # Hiển thị dân số dưới dạng số nguyên
print(f"- Độ lệch chuẩn: {summary['Population']['std']:.2f}")
print(f"- Giá trị nhỏ nhất: {int(summary['Population']['min']):,}")
print(f"- Phân vị thứ 25: {int(summary['Population']['25%']):,}")
print(f"- Trung vị: {int(summary['Population']['50%']):,}")
print(f"- Phân vị thứ 75: {int(summary['Population']['75%']):,}")
print(f"- Giá trị lớn nhất: {int(summary['Population']['max']):,}")

# Tỷ lệ lạm phát
print("\nTỷ lệ lạm phát(Inflation Rate (%)): ")
print(f"- Số lượng: {int(summary['Inflation Rate (%)']['count'])} quan sát")
print(f"- Trung bình: {summary['Inflation Rate (%)']['mean']:.2f}%")
print(f"- Độ lệch chuẩn: {summary['Inflation Rate (%)']['std']:.2f}")
print(f"- Giá trị nhỏ nhất: {summary['Inflation Rate (%)']['min']:.2f}%")
print(f"- Phân vị thứ 25: {summary['Inflation Rate (%)']['25%']:.2f}%")
print(f"- Trung vị: {summary['Inflation Rate (%)']['50%']:.2f}%")
print(f"- Phân vị thứ 75: {summary['Inflation Rate (%)']['75%']:.2f}%")
print(f"- Giá trị lớn nhất: {summary['Inflation Rate (%)']['max']:.2f}%")

# Năm
print("\nNăm(Year): ")
print(f"- Số lượng: {int(summary['Year']['count'])} quan sát")
print(f"- Trung bình: {summary['Year']['mean']:.2f}")
print(f"- Độ lệch chuẩn: {summary['Year']['std']:.2f}")
print(f"- Giá trị nhỏ nhất: {summary['Year']['min']:.2f}")
print(f"- Phân vị thứ 25: {summary['Year']['25%']:.2f}")
print(f"- Trung vị: {summary['Year']['50%']:.2f}")
print(f"- Phân vị thứ 75: {summary['Year']['75%']:.2f}")
print(f"- Giá trị lớn nhất: {summary['Year']['max']:.2f}")

forecast_df.to_csv(r'D:\STUDY\TTNT\ĐỒ ÁN\forecasted_unemployment.csv', index=False)
print("Kết quả dự đoán đã được lưu vào tệp 'forecasted_unemployment.csv'.")