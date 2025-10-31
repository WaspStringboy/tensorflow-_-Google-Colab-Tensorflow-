# tensorflow圖形檢測_使用Google Colab使用Tensorflow進行自定義對象檢測<br/>

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
![image](https://github.com/ojiver/AI/blob/main/1.jpg?raw=true)
