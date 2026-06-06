import os
import glob
from pathlib import Path
import cv2
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
        self.model = YOLO(model_path)
        self.spec_base_dir = spec_base_dir
        self.output_base_dir = output_base_dir
        self.bug_report_base_dir = "./bug_report"

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
        image_path: str, 
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
        results = self.model.predict(source=image_path, conf=0.25, verbose=False)
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
                # print(f"-> 捨棄過小目標 (比例不足): w_ratio={box_w_ratio:.4f}, h_ratio={box_h_ratio:.4f}")
                continue
                
            # 檢查條件 B: 是否小於指定的絕對像素門檻
            if box_w_pixel < min_w_pixel or box_h_pixel < min_h_pixel:
                # print(f"-> 捨棄過小目標 (像素不足): w_pixel={box_w_pixel}, h_pixel={box_h_pixel}")
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
    
    def save_bug_report(self, image_path: str, sorted_boxes: list, result_file: str):
        """
        新增模塊: 當數量不符時，將原圖與預測結果（劃上 Bounding Box）存入 bug_report 專屬資料夾
        """
        # 1. 建立對應的結果資料夾路徑，命名形式與結果相同
        bug_dir = os.path.join(self.bug_report_base_dir, result_file)
        os.makedirs(bug_dir, exist_ok=True)
        
        img_stem = Path(image_path).stem
        img_ext = Path(image_path).suffix  # 保留原始副檔名 (.jpg / .png)
        
        # 讀取原圖
        orig_img = cv2.imread(image_path)
        
        # 2. 儲存原圖片
        original_save_path = os.path.join(bug_dir, f"{img_stem}_orig{img_ext}")
        cv2.imwrite(original_save_path, orig_img)
        
        # 3. 繪製預測結果 (在複製的圖片上畫框與排序標號)
        viz_img = orig_img.copy()
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

    def apply_labels_and_save(self, image_path: str, sorted_boxes: list, spec_labels: list, output_dir: str, result_file: str): 
        """
        步驟 2: 進行貼標籤與儲存 (不立刻儲存，在這邊處理過濾與命名)
        """
        if not sorted_boxes:
            if len(spec_labels) > 0:
                print(f"[異常] 圖片 {os.path.basename(image_path)} 未偵測到任何物件，但 Spec 預期有 {len(spec_labels)} 個。")
                self.save_bug_report(image_path, sorted_boxes, result_file)
            return
            
        if len(sorted_boxes) != len(spec_labels):
            print(f"[異常] 圖片 {os.path.basename(image_path)} 偵測數量({len(sorted_boxes)}) 與 Spec 數量({len(spec_labels)}) 不符！")
            self.save_bug_report(image_path, sorted_boxes, result_file)      
            return
        
        # 讀取原始圖片
        orig_img = cv2.imread(image_path)
        img_stem = Path(image_path).stem
        
        # 同步遍歷「排序好的偵測框」與「Spec 定義的標籤順序」
        for index, (box_info, label_name) in enumerate(zip(sorted_boxes, spec_labels)):
            # 條件：若是 GND 則無視，保存 GND 以外的地方
            if label_name.upper() == "GND":
                continue
                
            x1, y1, x2, y2 = box_info["xyxy"]
            
            # 影像裁切
            cropped_img = orig_img[y1:y2, x1:x2]
            
            # 命名規則：{Spec定義的標籤名稱}.jpg
            output_filename = f"{label_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 儲存圖片
            cv2.imwrite(output_path, cropped_img)

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
            
        # 3. 建立儲存目的地資料夾: ./cropped_results/{Result_File}
        target_output_dir = os.path.join(self.output_base_dir, result_file)
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
            
            # A. 讀取該圖片對應的 Spec 標籤順序檔
            spec_labels = self.load_spec_labels(spec_name, img_stem)
            if not spec_labels:
                continue  # 找不到定義檔則跳過該張圖
                
            # B. 呼叫 Method 1: YOLO 偵測並由左至右排序 (不在此步驟儲存)
            sorted_boxes = self.crop_and_sort(
                img_path, 
                min_w_ratio=ratios["min_w"], 
                min_h_ratio=ratios["min_h"],
                max_w_ratio=ratios["max_w"],
                max_h_ratio=ratios["max_h"]
            )
            
            # C. 呼叫 Method 2: 傳入排序結果與 Spec 標籤進行貼標與儲存
            self.apply_labels_and_save(img_path, sorted_boxes, spec_labels, target_output_dir, result_file)
            print(f"   已完成圖片處理: {os.path.basename(img_path)}")
            
        print(f"\n[完成] 資料夾 {result_file} 處理完畢。")


# ==========================================
# 模塊實際執行範例
# ==========================================
if __name__ == "__main__":
    # 初始化處理器 (傳入您的 YOLO 模型路徑)
    processor = YOLOImageProcessor(
        model_path="best.pt", 
        spec_base_dir="./Spec", 
        output_base_dir="./cropped_results"
    )
    
    # 產線傳入的目標圖片資料夾位址 (支援 Windows 絕對路徑)
    target_folder = r"C:\Homework\YoloLabel_glue_clear\MCIO16i-MCIO16i\2H6113A3W13\WO-10000057032"
    folder = Path(target_folder)
    # 執行整個資料夾的自動化分析、排序、貼標籤與儲存
    count = 0
    for _ in folder.iterdir(): 
        count +=1
    file_count = count
    count = 1
    for item in folder.iterdir():
        print(f"[{count}/{file_count}]:")
        processor.process_folder(src_folder_path=item)
        count += 1

