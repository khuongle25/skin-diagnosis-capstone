import os
import requests
import zipfile

def get_onedrive_directdownload(share_url):
    if "1drv.ms" in share_url:
        resp = requests.head(share_url, allow_redirects=True)
        return resp.url.replace('redir?', 'download?')
    if "onedrive.live.com" in share_url:
        return share_url.replace("redir?", "download?")
    return share_url

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} đã tồn tại, bỏ qua tải lại.")
        return
    raw_url = get_onedrive_directdownload(url)
    print(f"Đang tải {save_path} từ {raw_url} ...")
    resp = requests.get(raw_url, stream=True)
    resp.raise_for_status()
    print(f"Đã tải xong {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)
    print(f"Đã tải xong {save_path}")

def extract_zip(zip_path, extract_to):
    print(f"Đang giải nén {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Đã giải nén vào {extract_to}")

def download_and_extract_checkpoints(zip_url, zip_save_path, checkpoints_dir):
    # Nếu đã có đủ model thì bỏ qua tải/giải nén
    if os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0:
        print(f"Thư mục {checkpoints_dir} đã có model, bỏ qua tải lại.")
        return
    download_file(zip_url, zip_save_path)
    extract_zip(zip_save_path, checkpoints_dir)
    print("Đã tải và giải nén model thành công")