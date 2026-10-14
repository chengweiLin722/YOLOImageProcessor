import os
import glob
import difflib
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO


class YOLOImageProcessor:

    def __init__(self, model_path: str, spec_base_dir: str = "./Spec", output_base_dir: str = "./cropped_results"):
        """
        初始化處理模塊
        :param model_path: YOLO .pt 模型路徑
        :param spec_base_dir: 規格定義檔的根目錄
        :param output_base_dir: 裁切結果的根目錄
        :param bug_report_base_dir: 錯誤輸出的根目錄
        """
        self.name = model_path
        self.model = YOLO(model_path)
        self.spec_base_dir = spec_base_dir
        self.output_base_dir = output_base_dir
        self.bug_report_base_dir = "./bug_report"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 效率統計器變數
        self.stats = {
            "total_images": 0,
            "yolo_time": 0.0,      # YOLO 推論純耗時 (秒)
            "crop_io_time": 0.0,   # 裁切、儲存耗時 (秒)
            "angle_detect": 0.0,   # 角度偵測到旋轉結束耗時 (秒)
            "load_image": 0.0,     # 圖片讀取耗時 (秒)
            "total_boxes": 0       # 總共成功裁切的焊點框數
        }

    def reset_stats(self):
        """
        重置效率統計計數器
        """
        self.stats = {
            "total_images": 0,
            "yolo_time": 0.0,
            "crop_io_time": 0.0,
            "angle_detect": 0.0,
            "load_image": 0.0,
            "total_boxes": 0
        }

    def parse_path_info(self, folder_path: str):
        """
        解析資料夾路徑以獲取 Spec 名稱與 Result_File 名稱
        例如輸入: C:\\Homework\\YoloLabel_precise_same\\MCIO8i-MCIO8i\\6H6112A3AG5\\20260129001\\SIG52601290001_20260130_170155
        將提取出:
          - spec_name: MCIO8i-MCIO8i
          - result_file: SIG52601290001_20260130_170155
        """
        # 標準化路徑分隔符
        pure_path = Path(folder_path)
        parts = pure_path.parts
        parts = parts[1:]
        # 尋找包含關鍵字或已知規格定義的資料夾名稱，這裡以範例結構從後往前推
        # Result_File 是最後一個 element
        result_file = parts[-1]
        
        # 尋找在 ./Spec 下確實存在的資料夾名稱
        spec_name = None
        for part in parts:
            if os.path.isdir(os.path.join(self.spec_base_dir, part)):
                spec_name = part
                break
                
        # 備用方案：如果沒匹配到，採用固定相對層級 (倒數第5層)
        if not spec_name and len(parts) >= 5:
            spec_name = parts[-5]
            
        if not spec_name:
            raise ValueError(f"無法在路徑中解析出有效的 Spec 名稱，請確認該 Spec 名稱資料夾是否存在於 {self.spec_base_dir} 中。")
            
        return spec_name, result_file

    def load_spec_labels(self, spec_name: str, img_stem: str) -> list:
        """
        讀取對應的 Spec txt 檔案，獲取標籤順序
        例如: ./Spec/MCIO8i-MCIO8i/Face1_Img1.txt
        """
        spec_txt_path = os.path.join(self.spec_base_dir, spec_name, f"{img_stem}.txt")
        
        if not os.path.exists(spec_txt_path):
            print(f"[警告] 找不到對應的 Spec 定義檔: {spec_txt_path}，該圖片將無法進行貼標籤步驟。")
            return []
            
        with open(spec_txt_path, 'r', encoding='utf-8') as f:
            # 讀取每一行，並去除前後空白與換行符
            labels = [line.strip() for line in f.readlines() if line.strip()]
        return labels

    def load_spec_ratio(self, spec_name: str) -> dict:
        """
        【新增方法】讀取 Spec 目錄下的 ratio.txt
        預期格式：
        min_w_ratio min_h_ratio
        max_w_ratio max_h_ratio
        """
        ratio_txt_path = os.path.join(self.spec_base_dir, spec_name, "ratio.txt")
        
        # 設定預設值 (防呆用：萬一某規格忘記放 ratio.txt，系統不至於崩潰)
        default_ratios = {
            "min_w": 0.0, "min_h": 0.0,
            "max_w": 1.0, "max_h": 1.0
        }
        
        if not os.path.exists(ratio_txt_path):
            print(f"[提示] 找不到比例定義檔: {ratio_txt_path}，將使用預設門檻 (min=0.0, max=1.0)")
            return default_ratios
            
        try:
            with open(ratio_txt_path, 'r', encoding='utf-8') as f:
                lines = [line.strip().split() for line in f.readlines() if line.strip()]
            
            if len(lines) < 2 or len(lines[0]) < 2 or len(lines[1]) < 2:
                print(f"[警告] {ratio_txt_path} 格式不正確，改用預設門檻。")
                return default_ratios
                
            return {
                "min_w": float(lines[0][0]),
                "min_h": float(lines[0][1]),
                "max_w": float(lines[1][0]),
                "max_h": float(lines[1][1])
            }
        except Exception as e:
            print(f"[錯誤] 讀取 ratio.txt 失敗 ({e})，使用預設門檻。")
            return default_ratios
        
    def crop_and_sort(
        self, 
        image: np.ndarray, 
        min_w_ratio: float = 0.0, 
        min_h_ratio: float = 0.0,
        max_w_ratio: float = 1.0, 
        max_h_ratio: float = 1.0,
        min_w_pixel: int = 0,
        min_h_pixel: int = 0
    ) -> list:
        """
        步驟 1: 對單張圖片進行 YOLO 預測，並篩選掉尺寸過小的框，最後將結果由左至右排序
        
        :param image_path: 圖片路徑
        :param min_w_ratio: 最小寬度比例門檻 (0.0 ~ 1.0)，低於此值則捨棄。例如設為 0.05
        :param min_h_ratio: 最小高度比例門檻 (0.0 ~ 1.0)，低於此值則捨棄。例如設為 0.10
        :param min_w_pixel: 最小寬度像素門檻 (單位: 像素)，低於此值則捨棄。
        :param min_h_pixel: 最小高度像素門檻 (單位: 像素)，低於此值則捨棄。
        :return: 篩選並排序後的偵測框資訊列表
        """
        if self.device == 'cuda':
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        results = self.model.predict(source=image, conf=0.1, verbose=False)

        if self.device == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        self.stats["yolo_time"] += (t_end - t_start)

        result = results[0]
        
        # 取得原始圖片的實際像素寬高 (Height, Width)
        img_h, img_w = result.orig_shape
        
        valid_boxes = []
        for box in result.boxes:
            # 1. 取得絕對像素座標 (xyxy: x_min, y_min, x_max, y_max)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # 計算該偵測框的實際像素寬高
            box_w_pixel = x2 - x1
            box_h_pixel = y2 - y1
            
            # 2. 計算該偵測框相對於整張圖的正規化寬高比例 (即您提供的 0.115625 與 0.352778 概念)
            box_w_ratio = box_w_pixel / img_w
            box_h_ratio = box_h_pixel / img_h
            
            # 【尺寸篩選邏輯】
            # 檢查條件 A: 是否小於指定的相對比例門檻
            if box_w_ratio < min_w_ratio or box_h_ratio < min_h_ratio or box_w_ratio > max_w_ratio or box_h_ratio > max_h_ratio:
                #print(f"-> 捨棄過小目標 (比例不足): w_ratio={box_w_ratio:.4f}, h_ratio={box_h_ratio:.4f}")
                continue
                
            # 檢查條件 B: 是否小於指定的絕對像素門檻
            if box_w_pixel < min_w_pixel or box_h_pixel < min_h_pixel:
                #print(f"-> 捨棄過小目標 (像素不足): w_pixel={box_w_pixel}, h_pixel={box_h_pixel}")
                continue
            
            # 符合所有尺寸條件才加入列表
            cls_id = int(box.cls[0].cpu().item())
            cls_name = self.model.names[cls_id]
            
            valid_boxes.append({
                "xyxy": xyxy,
                "yolo_cls": cls_name,
                "x_min": x1
            })
            
        # 根據 Bounding Box 的左側 X 座標由左至右排序
        valid_boxes.sort(key=lambda item: item["x_min"])
        return valid_boxes
    
    def save_bug_report(self, image_path: str, image: np.ndarray, sorted_boxes: list, result_file: str):
        """
        新增模塊: 當數量不符時，將原圖與預測結果（劃上 Bounding Box）存入 bug_report 專屬資料夾
        """
        # 1. 建立對應的結果資料夾路徑，命名形式與結果相同
        bug_dir = os.path.join(self.bug_report_base_dir, result_file)
        os.makedirs(bug_dir, exist_ok=True)
        
        img_stem = Path(image_path).stem
        img_ext = Path(image_path).suffix  # 保留原始副檔名 (.jpg / .png)
        
        
        # 2. 儲存原圖片
        original_save_path = os.path.join(bug_dir, f"{img_stem}_orig{img_ext}")
        cv2.imwrite(original_save_path, image)
        
        # 3. 繪製預測結果 (在複製的圖片上畫框與排序標號)
        viz_img = image.copy()
        for index, box_info in enumerate(sorted_boxes):
            x1, y1, x2, y2 = box_info["xyxy"]
            cls_name = box_info["yolo_cls"]
            
            # 畫綠色邊界框
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框上方寫上 [由左至右順序_YOLO類別] 方便除錯
            label_text = f"#{index+1} {cls_name}"
            cv2.putText(viz_img, label_text, (x1, max(y1 - 5, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # 4. 儲存預測結果圖片
        pred_save_path = os.path.join(bug_dir, f"{img_stem}_pred{img_ext}")
        cv2.imwrite(pred_save_path, viz_img)
        
        print(f"   [Bug Report] 已將異常圖片與預測結果匯出至: {bug_dir}")

    def apply_labels_and_save(self, image_path: str, image: np.ndarray, sorted_boxes: list, spec_labels: list, output_dir: str, result_file: str): 
        """
        步驟 2: 進行貼標籤與儲存 (不立刻儲存，在這邊處理過濾與命名)
        """
        t_start = time.perf_counter()

        if not sorted_boxes:
            if len(spec_labels) > 0:
                print(f"[異常] 圖片 {os.path.basename(image_path)} 未偵測到任何物件，但 Spec 預期有 {len(spec_labels)} 個。")
                self.save_bug_report(image_path, image, sorted_boxes, result_file)
            return
            
        if len(sorted_boxes) != len(spec_labels):
            print(f"[異常] 圖片 {os.path.basename(image_path)} 偵測數量({len(sorted_boxes)}) 與 Spec 數量({len(spec_labels)}) 不符！")
            self.save_bug_report(image_path, image, sorted_boxes, result_file)      
            return
        
        # 同步遍歷「排序好的偵測框」與「Spec 定義的標籤順序」
        for index, (box_info, label_name) in enumerate(zip(sorted_boxes, spec_labels)):
            # 條件：若是 GND 則無視，保存 GND 以外的地方
            if label_name.upper() == "GND":
                continue
                
            x1, y1, x2, y2 = box_info["xyxy"]
            
            # 影像裁切
            cropped_img = image[y1:y2, x1:x2]
            
            # 命名規則：{Spec定義的標籤名稱}.jpg
            output_filename = f"{label_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 儲存圖片
            cv2.imwrite(output_path, cropped_img)
            self.stats["total_boxes"] += 1

        t_end = time.perf_counter()
        self.stats["crop_io_time"] += (t_end - t_start)

    def detect_pcb_angle(self, image: np.ndarray, max_angle_threshold: float = 30.0) -> float:
        """
        透過邊緣檢測與霍夫直線變換抓取 PCB 主板邊，計算傾斜角 (度)
        :param image: 輸入彩色或灰階影像
        :param max_angle_threshold: 允許的最大傾斜角 (避免誤抓垂直走線)
        :return: 傾角 (度)，若未找到合適邊緣則回傳 0.0
        """
        h, w = image.shape[:2]
    
        # 1. 核心優化：等比例縮小長邊至 800px 進行快速幾何計算 (節省 90% 運算量)
        max_dim = 400
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small_w = int(w * scale)
            small_h = int(h * scale)
            # 使用 INTER_NEAREST 或 INTER_LINEAR 快速縮小
            small_img = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_img = image

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        # 輕微高斯模糊，降低 PCB 表面錫點與文字雜訊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny 邊緣檢測 (門檻值可依板材對比度微調)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # 霍夫直線檢測 (抓取長度大於影像寬度 15% 的線段)
        min_line_len = int(min(small_img.shape[:2]) * 0.20)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_line_len, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 計算線段長度與角度
            length = np.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # 將角度正規化到 [-45, 45] 範圍內 (針對接近水平的主板邊)
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90

            if abs(angle) <= max_angle_threshold:
                # 權重採用線段長度，愈長的板邊可信度愈高
                angles.append((angle, length))

        if not angles:
            return 0.0

        # 依長度加權計算平均角度
        total_len = sum(l for _, l in angles)
        weighted_angle = sum(a * l for a, l in angles) / total_len
        return float(-weighted_angle)

    def process_folder(self, src_folder_path: str):
        """
        主調用模塊：處理目標資料夾內的所有圖片
        """
        print(f"開始解析資料夾路徑: {src_folder_path}")
        
        # 1. 解析路徑獲取 Spec 與 Result_File
        try:
            spec_name, result_file = self.parse_path_info(src_folder_path)
            print(f"-> 解析成功! Spec: {spec_name} | Result_File: {result_file}")
        except Exception as e:
            print(f"-> 錯誤: {e}")
            return
        
        # 2. 讀取spec尺寸
        ratios = self.load_spec_ratio(spec_name)
        print(f"-> 成功加載尺寸配置: Min({ratios['min_w']}, {ratios['min_h']}) | Max({ratios['max_w']}, {ratios['max_h']})")
            
        # 3. 建立儲存目的地資料夾: ./cropped_results/{spec_name}/{Result_File}
        # 取得 output_base_dir 下所有現有的子資料夾清單
        existing_dirs = []
        if os.path.exists(self.output_base_dir):
            existing_dirs = [
                d for d in os.listdir(self.output_base_dir) 
                if os.path.isdir(os.path.join(self.output_base_dir, d))
            ]

        # 使用 difflib 尋找最相近的資料夾名稱 (cutoff 可調整相似度門檻，0.0 ~ 1.0)
        matches = difflib.get_close_matches(spec_name, existing_dirs, n=1, cutoff=0.5)

        if matches:
            matched_spec_name = matches[0]
            print(f" 找到最相近的 Spec 資料夾: '{spec_name}' -> '{matched_spec_name}'")
        else:
            matched_spec_name = spec_name
            print(f" 未找到相近資料夾，將直接建立新 Spec 資料夾: '{matched_spec_name}'")

        target_output_dir = os.path.join(self.output_base_dir, matched_spec_name, result_file)
        os.makedirs(target_output_dir, exist_ok=True)
        print(f"-> 裁切影像將儲存至: {target_output_dir}")        

        # 4. 獲取資料夾內所有圖片（支援 jpg, jpeg, png, bmp，自動忽略 txt）
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(src_folder_path, ext)))
            
        if not image_files:
            print("-> 找不到任何支援的圖片檔案。")
            return
            
        print(f"-> 找到 {len(image_files)} 張圖片，開始進行 YOLO 預測與 Spec 貼標籤...")
        
        # 5. 開始循環處理每張照片
        for img_path in image_files:
            img_stem = Path(img_path).stem  # 例如: Face1_Img1

            t_start = time.perf_counter()

            img = cv2.imread(img_path)

            t_end = time.perf_counter()
            self.stats["load_image"] += (t_end - t_start)

            if img is None:
                print(f"[警告] 讀取失敗，img 為 None，請檢查路徑是否存在: {img_path}")
                continue


            # A. 偵測角度 (輕量)
            t_start = time.perf_counter()

            detected_angle = self.detect_pcb_angle(img)
            print(f"偵測傾角: {detected_angle:+6.2f}°")
            # B. 條件式轉正 (只有歪斜顯著時才花時間做 warpAffine)
            ANGLE_THRESHOLD = 1  # 容許門檻：1 度以內不轉

            if abs(detected_angle) > ANGLE_THRESHOLD:
                print("偵測到傾斜圖片，正在進行調整...")
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), -detected_angle, 1.0)
                process_img = cv2.warpAffine(
                    img, 
                    M, 
                    (w, h), 
                    flags=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REPLICATE
                )
            else:
                process_img = img

            t_end = time.perf_counter()
            self.stats["angle_detect"] += (t_end - t_start)

            # C. 讀取該圖片對應的 Spec 標籤順序檔
            spec_labels = self.load_spec_labels(spec_name, img_stem)
            if not spec_labels:
                continue  # 找不到定義檔則跳過該張圖
                
            # D. 呼叫 Method: YOLO 偵測並由左至右排序 (不在此步驟儲存)
            sorted_boxes = self.crop_and_sort(
                process_img, 
                min_w_ratio=ratios["min_w"], 
                min_h_ratio=ratios["min_h"],
                max_w_ratio=ratios["max_w"],
                max_h_ratio=ratios["max_h"]
            )
            
            # E. 呼叫 Method: 傳入排序結果與 Spec 標籤進行貼標與儲存
            self.apply_labels_and_save(img_path, process_img, sorted_boxes, spec_labels, target_output_dir, result_file)
            self.stats["total_images"] += 1
            print(f"   已完成圖片處理: {os.path.basename(img_path)}")
            
        print(f"\n[完成] 資料夾 {result_file} 處理完畢。")

    def print_efficiency_report(self, total_wall_time: float):
            """
            輸出整體運行效率統計分析報表
            """
            n_img = self.stats["total_images"]
            if n_img == 0:
                print("\n 未處理任何圖片，無效率數據。")
                return

            yolo_t = self.stats["yolo_time"]
            crop_t = self.stats["crop_io_time"]
            load_t = self.stats["load_image"]
            angle_t = self.stats["angle_detect"]
            other_t = max(0.0, total_wall_time - (yolo_t + crop_t + angle_t + load_t))
            fps = n_img / total_wall_time

            print("\n" + "=" * 65)
            print(" YOLO 影像處理與裁切管線 - 整體效率分析報告")
            print(f" 使用權重：{self.name}")
            print("=" * 65)
            print(f"  - 處理總圖片張數:        {n_img} 張")
            print(f"  - 成功裁切焊點總數:      {self.stats['total_boxes']} 個")
            print(f"  - 總端到端耗時 (Wall):   {total_wall_time:.3f} 秒")
            print(f"  - 整體處理吞吐量:        {fps:.2f} 張/秒 (FPS)")
            print(f"  - 單張大圖平均總耗時:    {(total_wall_time / n_img) * 1000:.2f} ms")
            print("-" * 65)
            print(" 各階段耗時細項拆解 (單張平均):")
            print(f"  1. YOLO 偵測推論:        {(yolo_t / n_img) * 1000:7.2f} ms | 佔比: {(yolo_t / total_wall_time) * 100:5.1f}%")
            print(f"  2. OpenCV 裁切與儲存:    {(crop_t / n_img) * 1000:7.2f} ms | 佔比: {(crop_t / total_wall_time) * 100:5.1f}%")
            print(f"  2. OpenCV 讀取:         {(load_t / n_img) * 1000:7.2f} ms | 佔比: {(load_t / total_wall_time) * 100:5.1f}%")
            print(f"  4. 角度偵測、旋轉:       {(angle_t / n_img) * 1000:7.2f} ms | 佔比: {(angle_t / total_wall_time) * 100:5.1f}%")
            print(f"  5. 路徑解析與其它開銷:   {(other_t / n_img) * 1000:7.2f} ms | 佔比: {(other_t / total_wall_time) * 100:5.1f}%")
            print("=" * 65)

# ==========================================
# 模塊實際執行範例
# ==========================================
if __name__ == "__main__":
    # 初始化處理器 (傳入您的 YOLO 模型路徑)
    processor = YOLOImageProcessor(
        model_path="yolov8n.pt", 
        spec_base_dir="./Spec", 
        output_base_dir="./cropped_results_test"
    )
    
    # 產線傳入的目標圖片資料夾位址 (支援 Windows 絕對路徑)
    target_folder = r"C:"
    folder = Path(target_folder)
    # 執行整個資料夾的自動化分析、排序、貼標籤與儲存
    folders_to_process = [item for item in folder.iterdir() if item.is_dir()]
    file_count = len(folders_to_process)

    print(f" 開始執行批次裁切作業，預計處理 {file_count} 個子資料夾...")
    overall_start_time = time.perf_counter()

    for count, item in enumerate(folders_to_process, 1):
        print(f"\n[{count}/{file_count}]:")
        processor.process_folder(src_folder_path=str(item))

    overall_end_time = time.perf_counter()
    total_wall_time = overall_end_time - overall_start_time

    processor.print_efficiency_report(total_wall_time)
