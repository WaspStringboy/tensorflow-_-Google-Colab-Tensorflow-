# tensorflow圖形檢測_使用Google Colab使用Tensorflow進行自定義對象檢測<br/>

![image](1.jpeg)
在此文章中，我們將使用 Tensorflow 物件偵測 (Object Detection) API 來建立自定義的物件偵測器。我將選擇偵測蘋果果實，但是您可以選擇要偵測自己的自定義對象的任何圖像。<br/>

步驟：<br/>
安裝 (Installation)<br/>
收集資料 (Gathering data)<br/>
標註資料 (Labeling data)<br/>
生成 TFRecords 以供訓練 (Generating TFRecords for training)<br/>
配置訓練 (Configuring training)<br/>
訓練模型 (Training model)<br/>
匯出推論圖 (Exporting inference graph)<br/>
測試物件偵測器 (Testing object detector)<br/>
一、安裝 (Installation)<br/>
1.Python 3.6 或更高版本。<br/>
2.Ubuntu 18.04 / Google Colab。<br/>
3.Tensorflow / Tensorflow-gpu。<br/>
4.克隆 Tensorflow 模型儲存庫。<br/>
![image](2.png)<br/>
1. 檢查環境<br/>

Protobuf 編譯：TensorFlow 物件偵測 API 使用 Protobufs 來配置模型與訓練參數。在使用該框架之前，必須先編譯 Protobuf 檔案。<br/>
這可以透過從 tensorflow/models/research/ 目錄運行以下命令來完成：<br/>
![image](3.png)<br/>
將系統路徑加入 PYTHONPATH<br/>
在 Google Colab 上運行時，應將 Tfmodels/research 和 slim 目錄新增到 PYTHONPATH。<br/>
Object Detection Installation（安裝物件偵測）<br/>
Testing the Installation（測試安裝）<br/>
使用 Google Colab 的範例可參閱下方連結：<br/>
使用 Google Colab 訓練 Tensorflow 物件偵測 API<br/>
![image](4.png)<br/>
二、資料收集（Gathering data）<br/>
2.1<br/>
打開 Google Chrome 瀏覽器，搜尋並安裝一個名為 Download All Images 的瀏覽器擴充套件。<br/>
![image](5.png)<br/>
2.2<br/>
現在在 Google 上輸入並搜尋你想要的對象，例如「Apple」，點擊「下載所有圖像」的擴充套件按鈕。這樣就能批次下載圖片，通常會自動儲存為一個 .zip 壓縮檔。<br/>
![image](6.png)<br/>
三、資料標註（Labeling data）<br/>
打開終端機並輸入以下方式安裝 LabelImg<br/>
LabelImg 是用於影像標註的工具。<br/>
安裝 LabelImg 後，透過這個指令來啟動：<br/>
![image](7.png)<br/>
在不同的環境中安裝 LabelImg 的方法可能不同，可以參考以下網站：<br/>
👉 https://github.com/tzutalin/labelImg<br/>
上面的內容並非所有圖片均完成標註，它正在進行中。<br/>
LabelImg 在每張圖像旁會生成一個 XML 文件，裡面包含了物件名稱與邊界框的座標資訊。<br/>
這裡大約有 100 張圖片。<br/>
現在需要克隆儲存庫：<br/>
👉 https://github.com/zjgulai/Tensorflow-Object-Detection-API-With-Custom-Dataset<br/>

使用以下命令克隆：<br/>
完成後進入該目錄即可進行後續操作。<br/>








