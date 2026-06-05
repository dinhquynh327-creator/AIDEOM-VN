FROM python:3.10-slim

# Ngăn Python ghi file bytecode (.pyc)
ENV PYTHONDONTWRITEBYTECODE 1
# Hiển thị log ngay lập tức
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Cài đặt các thư viện hệ thống cơ bản và một số solver thường dùng cho pyomo/pulp
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    glpk-utils \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt requirements
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy mã nguồn
COPY . /app/
