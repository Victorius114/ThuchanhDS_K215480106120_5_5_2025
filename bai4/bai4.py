import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.stats import linregress

'''Bài 4: Phân tích dữ liệu môi trường từ cảm biến IoT
Mô tả: Giả định bạn làm việc cho một công ty môi trường, thu thập dữ liệu từ các cảm biến IoT đo chất lượng không khí tại một thành phố. Dữ liệu bao gồm: thời gian, vị trí (tọa độ x, y), nồng độ PM2.5, nhiệt độ, độ ẩm, và tốc độ gió. Dữ liệu này không có sẵn và bạn cần mô phỏng.
Dữ liệu:https://drive.google.com/file/d/1_UKvFZwRSb8gwqNnAkz2pGmgGnZOq6nO/view?usp=sharing
Yêu cầu:
+ Sử dụng thuật toán DBSCAN để phân cụm các khu vực dựa trên tọa độ và nồng độ PM2.5, xác định các "điểm nóng ô nhiễm".
+ Tính toán chỉ số "rủi ro ô nhiễm" cho mỗi khu vực dựa trên PM2.5 trung bình và tần suất vượt ngưỡng an toàn (giả định ngưỡng là 50 µg/m³).
+ Tạo đặc trưng "chỉ số thời tiết bất lợi" bằng cách kết hợp nhiệt độ, độ ẩm, và tốc độ gió (tự định nghĩa công thức, ví dụ: độ ẩm cao + gió yếu = bất lợi).
+ Tạo đặc trưng "xu hướng ô nhiễm" bằng cách tính độ dốc của PM2.5 trong 24 giờ trước đó cho mỗi vị trí.
+ Xây dựng mô hình LSTM (Long Short-Term Memory) để dự đoán nồng độ PM2.5 trong 6 giờ tới tại một vị trí cụ thể, sử dụng dữ liệu lịch sử (24 giờ trước).
+ Đánh giá mô hình bằng RMSE và vẽ biểu đồ so sánh giá trị thực tế và dự đoán.'''

df = pd.read_csv(r'Data_Number_4.csv')

menu = 0
while True:
    print("Bài 4")
    print("1. Sử dụng thuật toán DBSCAN để phân cụm các khu vực dựa trên tọa độ và nồng độ PM2.5, xác định các điểm nóng ô nhiễm")
    print("2. Tính toán chỉ số rủi ro ô nhiễm cho mỗi khu vực dựa trên PM2.5 trung bình và tần suất vượt ngưỡng an toàn (giả định ngưỡng là 50 µg/m³).")
    print("3. Tạo đặc trưng chỉ số thời tiết bất lợi bằng cách kết hợp nhiệt độ, độ ẩm, và tốc độ gió (tự định nghĩa công thức, ví dụ: độ ẩm cao + gió yếu = bất lợi)")
    print("4. Tạo đặc trưng xu hướng ô nhiễm bằng cách tính độ dốc của PM2.5 trong 24 giờ trước đó cho mỗi vị trí.")
    print("5. Xây dựng mô hình LSTM (Long Short-Term Memory) để dự đoán nồng độ PM2.5 trong 6 giờ tới tại một vị trí cụ thể, sử dụng dữ liệu lịch sử (24 giờ trước)." \
    "\nĐánh giá mô hình bằng RMSE và vẽ biểu đồ so sánh giá trị thực tế và dự đoán.")
    a = input("Nhập lựa chọn: ")
    match a:
        case "1":
            X = df[['x_coord', 'y_coord', 'pm25']].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            dbscan = DBSCAN(eps = 0.5, min_samples = 5)
            clusters = dbscan.fit_predict(X_scaled)
            df['clusters'] = dbscan.fit_predict(X_scaled)
            df['clusters'] = clusters
            hotspots = df[df['clusters'] != 1].groupby('clusters')['pm25'].mean().sort_values(ascending=False)
            print("Các cụm ô nhiễm cao nhất: ")
            print(hotspots)

        case "2":
            nguong = 50
            df['vuot_muc'] = df['pm25'] > nguong

            riskdf = df.groupby(['x_coord', 'y_coord']).agg(
                pm25 = ('pm25', 'mean'),
                vuot_muc = ('vuot_muc', 'mean')
            ).reset_index()

            #Chỉ số rủi ro: risk = pm25.mean() * ngưỡng
            riskdf['risk_score'] = riskdf['pm25'] * riskdf['vuot_muc']
            riskdf = riskdf.sort_values(by = 'risk_score', ascending = False)
            print(riskdf.head())

        case "3":
            df['bat_loi'] = (
                0.5 * df['humidity'] + 
                (30 - df['temperature']).clip(lower=0) * 0.3 +
                (2 - df['wind_speed']).clip(lower=0) * 0.2
            )
            print(df['bat_loi'].head())
        
        case "4":
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values(by=['x_coord', 'y_coord', 'timestamp'], inplace=True)

            # Gộp theo vị trí
            trend_list = []

            for (x, y), group in df.groupby(['x_coord', 'y_coord']):
                group = group.copy()
                group['pm25_trend'] = group['pm25'].diff()  # đơn giản: độ chênh giữa giờ hiện tại và giờ trước
                trend_list.append(group)

            df = pd.concat(trend_list)
            
            print("Đã tính xu hướng ô nhiễm (pm25_trend) bằng cách lấy hiệu số giờ trước.")
            print(df['pm25_trend'])

        case "5":

        case _:
            False

    cont = input("\nBạn có muốn tiếp tục không? (y/n): ")
    if cont.lower() == 'n':
        break