# 資料與介面

## 特徵 schema V1

每個區段輸出九項非負特徵：點數、相對質心的 RMS 距離、感測距離差、圓擬合殘差平方和、半徑、邊界長度、相鄰點距離的母體變異數、平均曲率、平均絕對轉角。這套定義統一訓練與推論，不沿用舊 MATLAB 模型參數。

圓擬合以中心化座標進行帶 column pivoting 的 QR。秩不足、無有效半徑或半徑超過 100 m 時，以 100 m 半徑 sentinel 和總位置離散量表示退化特徵。位置輸出始終是區段質心，並非圓心。分類器對九項特徵各自取 log1p，再估計 Gaussian NB；變異數下限 1e-9。模型 magic 為 `LEG_NB_LOG1P_V3`。

## CLI

以空白分欄，每筆一列。拒絕截斷記錄。

| 命令 | 輸入 |
|---|---|
| `features segments.txt` | id label point_count x1 y1 x2 y2 ... |
| `train features.txt model.txt` | id label f0 ... f8；label 0 非腿、1 腿 |
| `classify features.txt model.txt` | 同上；輸出 label、log-score、prediction CSV |
| `track trajectory.txt` | stamp true_x true_y detection_count x1 y1 ...；true 座標只供輸出比較，不進入估測 |
| `scan scans.txt model.txt` | stamp angle_min angle_increment range_min range_max point_count ranges... |

掃描文字範例以 0 表示無效／無回波；ROS 介面另處理 NaN、Inf。分類使用扣除已校準門檻後的 log-score > 0。模型第二行保存門檻；train 可加上第四個 validation_features 路徑以選擇 F1 最大的門檻。Gaussian score 不宣稱是校準後的可信機率。

## ROS 時間與座標

`/scan` 的來源 frame 依時間戳轉到 `tracking_frame`（預設 odom）。量測及估計位置使用公尺，時間使用秒，角度使用弧度。命令輸出前轉回 `base_frame`（預設 base_link）。TF 不可用時停止預覽速度。

控制以 WallTimer 每 0.05 秒檢查狀態，ROS timestamp 與接收牆鐘皆要求資料不超過 0.3 秒。rosbag 暫停也會停止速度。倒序時間戳不會更新追蹤器；回放重啟需重啟節點。沒有掃描內各雷射束的運動去畸變，快速旋轉的誤差仍需另處理。

## 原始資料轉換

`tools/import_course_data.py` 讀取原課程兩個目錄的 point.mat 與 seg.mat，沿用其區段／類別標註，再輸出可交給 C++ 的點序列。公開庫包含重新計算的特徵與來源 SHA-256，沒有包含整份課程資料、底盤驅動或其他課程檔案。

不同序列的同一幀若逐位元完全相同會被排除；這無法排除近似影格或相同場景。歷史評估使用原區段標註，評估的是「已分段區段的分類」，不等於完整掃描偵測的 precision／recall。

