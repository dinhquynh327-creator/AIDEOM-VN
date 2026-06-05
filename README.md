# AIDEOM-VN Dashboard

Hệ thống **AIDEOM-VN Dashboard** là một ứng dụng Web tương tác được xây dựng bằng **Streamlit** (Frontend) và các thư viện Tối ưu hóa/Phân tích Dữ liệu mạnh mẽ của Python (Backend: `pulp`, `scipy`, `gymnasium`, v.v.). Hệ thống này giúp trực quan hóa và giải quyết 12 bài toán mô hình ra quyết định khác nhau (từ Quy hoạch tuyến tính, TOPSIS, đến Học tăng cường).

## Yêu cầu Hệ thống
Bạn **không cần** cài đặt Python, Node.js hay bất kỳ môi trường phức tạp nào trên máy tính. Chỉ cần:
1. **Docker Desktop** (hoặc Docker Engine).
2. **Visual Studio Code (VS Code)** (kèm extension *Dev Containers* hoặc *Docker* nếu cần).

## Hướng dẫn Chạy (Run) bằng Docker

Dự án đã được cấu hình sẵn môi trường chuẩn chỉnh (Containerized) thông qua file `Dockerfile` và `docker-compose.yml`. Bạn chỉ cần làm theo 2 bước cực kỳ đơn giản sau:

### Bước 1: Khởi động hệ thống
1. Mở VS Code, mở thư mục dự án này.
2. Mở Terminal trong VS Code (`Ctrl + ~`).
3. Gõ lệnh sau để Docker tự động tải môi trường, cài đặt thư viện và khởi chạy ứng dụng:
   ```bash
   docker-compose up --build
   ```
   *(Lưu ý: Quá trình `--build` trong lần chạy đầu tiên có thể mất vài phút do Docker cần tải Python và biên dịch một số gói thư viện toán học).*

### Bước 2: Truy cập ứng dụng
Sau khi Terminal báo `Uvicorn running on http://0.0.0.0:8501`, hãy mở trình duyệt web của bạn và truy cập vào đường link:

👉 **[http://localhost:8501](http://localhost:8501)**

Mọi thứ đã sẵn sàng! Giao diện Premium Dashboard với Sidebar điều hướng mượt mà sẽ xuất hiện.

---

## 🛠 Cách lập trình (Development) với VS Code + Docker

Hệ thống được cấu hình Volume Mount trong `docker-compose.yml` (`volumes: - .:/app`). Điều này mang lại một sức mạnh tuyệt vời:
- **Hot-Reload:** Bất kỳ thay đổi nào bạn thực hiện trên code (ví dụ sửa file `app.py` hay `src/optimization.py`) trong VS Code sẽ lập tức được tự động đồng bộ vào Container.
- Trình duyệt sẽ tự động phát hiện thay đổi và làm mới (refresh) giao diện mà bạn **không cần** phải khởi động lại lệnh `docker-compose up` hay build lại image!

### Cách tắt hệ thống
Khi không còn sử dụng, bạn hãy quay lại Terminal đang chạy lệnh trên và bấm tổ hợp phím `Ctrl + C` để dừng server, hoặc gõ lệnh:
```bash
docker-compose down
```

## Cấu trúc Dự án (Project Structure)
- `app.py`: Tệp tin chạy giao diện Streamlit (Frontend).
- `src/`: Chứa các thuật toán xử lý tối ưu, dữ liệu (Backend).
- `data/`: Các file dữ liệu CSV/Excel đầu vào.
- `Dockerfile` & `docker-compose.yml`: Cấu hình môi trường Containerized hoàn chỉnh.
