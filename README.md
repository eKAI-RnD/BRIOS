# BRIOS - Tái tạo dữ liệu bị che phủ bởi mây

## Cài đặt môi trường
- Bước 1: Cài đặt môi trường ảo (virtual environment):
    `python3 -m venv <tên môi trường>`
- Bước 2: Activate môi trường vừa cài đặt:
    `source <tên môi trường>/bin/activate`
- Bước 3: Cài đặt các thư viện cần dùng:
    `pip install -r requirements.txt`

## Các bước xử lý dữ liệu
- Bước 1: Chuẩn bị bộ dữ liệu
- Bước 2: Tiền xử lý dữ liệu:
  `python3 preprocessing/preprocess_for_<task>`
  task ở đây ví dụ như: LST_train (LST Train), Train (NDVI Train), Test (NDVI Inference)
  
## Huấn luyện mô hình
- Chạy toàn bộ pipeline:
    `bash BRIOS/run.sh`


## Cấu trúc các file 
- `models`: Chứa các file implement mô hình BRIOS. Ngoài ra còn chứa các file models, logs, history
- `notebooks`: Chứa các notebooks để thử nghiệm code
- `postprocessing`: Chứa postprocessing.py để hậu xử lý kết quả đầu ra
- `preprocessing`: Chứa các file tiền xử lý dữ liệu
- `results`: Chứa các file kết quả đầu ra của mô hình khi inference
- `supports`: Chứa các hàm hỗ trợ như earlystopping, ...

## Quy trình sửa đổi code khi cần 
- Bước 1: Tạo nhánh mới `git checkout -b <tên nhánh>`
- Bước 2: Thực hiện quá trình cải tiến code theo yêu cầu
- Bước 3: Thực hiện commit code lên nhánh vừa tạo
- Bước 4: Vào github nơi chứa project, tạo Pull Request với nhánh chính, ghi rõ mô tả các thay đổi về code
- Bước 5: Sau khi check code hoàn toàn đúng logic thì sẽ merge vào nhánh chính
