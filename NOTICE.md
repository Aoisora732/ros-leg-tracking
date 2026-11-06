# 來源與發布範圍

專案背景為 108 學年度第一學期「機器人感知估測」課程的期末專案，延續原有感知、分類、狀態估計與跟隨架構。公開版本為後續整理的 C++ 模組、測試、資料轉換工具與報告。

原課程套件含有課程範例與底盤環境相依內容，授權未完整標明；本庫不散布原套件、HyphaROS 驅動、其他課程程式或第三方原始資料包。歷史特徵由保留的課程掃描與標註重新計算；模型與結果不等於已通過實機驗證。

Eigen、GoogleTest、ROS、NumPy、Matplotlib 依各自授權使用；原始碼由套件管理器或建置工具取得，其授權不受本庫變更。本庫公開供查閱，目前未另授予通用開源授權。

技術參考（查閱日期 2026-11-07）：

- Eigen least squares: https://eigen.tuxfamily.org/dox/group__LeastSquares.html
- Gaussian Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Welch and Bishop, An Introduction to the Kalman Filter: https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
- ROS rostest: https://index.ros.org/p/rostest/
- ROS Noetic end of life: https://discourse.ros.org/t/ros-noetic-end-of-life-may-31-2025/43160
