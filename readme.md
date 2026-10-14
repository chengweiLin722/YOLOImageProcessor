# YOLO PCB 焊點自動化光學檢測 (AOI) 與裁切管線

本專案提供高效能的自動化 PCB 影像處理管線，整合 **輕量級幾何傾角校正 (Deskew)**、**YOLO 物件偵測**、**幾何尺寸篩選** 與 **Spec 定義標籤排序裁切**，適用於產線即時 (In-line / Near Real-Time) 焊點檢測。

## 重要環境安裝提示 (必讀)

> **注意：專案根目錄下的 `requirements.txt` 僅供套件清單與版本參考。**
>
> **請勿直接執行 `pip install -r requirements.txt`！**
>
> 若直接透過 pip 安裝，PyTorch 預設會下載 CPU 版本，導致 YOLO 推論無法使用 GPU 加速，大幅拖慢檢測吞吐量。請務必嚴格遵循下方的 環境安裝與啟動流程 逐步操作。

## 環境安裝與啟動流程

### 1. 建立獨立虛擬環境 (擇一)

#### 選項 A：使用 Conda 

```
conda create -n aoi_pipeline python=3.10 -y
conda activate aoi_pipeline

```

#### 選項 B：使用 Python 內建 venv

* **Windows (PowerShell / CMD)**:

  ```
  python -m venv aoi_env
  aoi_env\Scripts\activate
  
  ```

* **Linux / macOS**:

  ```
  python3 -m venv aoi_env
  source aoi_env/bin/activate
  
  ```

### 2. 安裝 GPU 支援之 PyTorch

請依照主機 NVIDIA 驅動支援的 CUDA 版本安裝 PyTorch（以 CUDA 12.1 為例）：

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

```

*(若顯卡或驅動較舊，可將 `cu121` 替換為 `cu118`)*

### 3. 安裝其餘相依套件

```
pip install ultralytics opencv-python numpy

```

### 4. 環境正確性測試

安裝完畢後，請在終端機中執行以下單行驗證指令：

```
python -c "import torch, cv2, ultralytics; print('============================='); print('CUDA GPU 加速可用:', torch.cuda.is_available()); print('GPU 裝置名稱:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '無'); print('OpenCV 版本:', cv2.__version__); print('Ultralytics 版本:', ultralytics.__version__); print('=============================')"

```

**驗證標準：**

* `CUDA GPU 加速可用` 必須顯示為 **`True`**。

* 若顯示為 `False`，代表 PyTorch 未正確吃到 GPU，請確認本機 NVIDIA 驅動程式並重裝對應版本的 PyTorch。

## 程式啟動方式

1. 確認工作目錄包含以下結構：

   ```
   .
   ├── YOLOImageProcessor.py   # 主程式
   ├── yolov8n.pt              # 模型權重檔
   ├── Spec/                   # 規格定義資料夾 (包含各 Spec 的 txt 與 ratio.txt)
   └── cropped_results_test/   # 輸出資料夾 (執行後自動建立)
   
   ```

2. 檢查 `YOLOImageProcessor.py` 底部 `__main__` 中的目標路徑設定：

   ```
   target_folder = r"C:\Your\Target\Image\Directory"
   
   ```

3. 執行主程式：

   ```
   python YOLOImageProcessor.py
   
   ```

## 系統架構與邏輯概述

整體管線採用一體化記憶體流程，避免大圖重複讀寫硬碟，確保極致吞吐量：

```
讀取原始影像 (cv2.imread / RAM Buffer)
       │
       ▼
[快速傾角偵測 (detect_pcb_angle)]
  ├─ 縮圖降採樣 (Max 400px，提速 90%)
  ├─ Canny 邊緣與霍夫直線 (HoughLinesP)
  └─ 傾斜角度加權統計
       │
       ▼
[條件式原地轉正 (Deskew)]
  ├─ |傾角| <= 1.0° ──► 略過旋轉，保持 100% 原始銳利度與 0 延遲
  └─ |傾角| >  1.0° ──► cv2.warpAffine 原地旋轉 (保持原圖尺寸，不削切邊緣焊點)
       │
       ▼
[YOLOv8 推論與尺寸過濾 (crop_and_sort)]
  ├─ RAM 矩陣直接推論 (略過二次磁碟 I/O)
  ├─ 依 Spec ratio.txt 過濾過大/過小雜訊框
  └─ 偵測框依 X 軸由左至右嚴格排序
       │
       ▼
[規格比對與裁切儲存 (apply_labels_and_save)]
  ├─ 數量一致 ──► 依 Spec 標籤命名並裁切焊點 (自動略過 GND)
  └─ 數量異常 ──► 觸發 save_bug_report 輸出標記圖以供人工複檢
       │
       ▼
[輸出各階段效能拆解報表 (print_efficiency_report)]

```