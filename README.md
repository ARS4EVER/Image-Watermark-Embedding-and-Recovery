# Image-Watermark-Embedding-and-Recovery
图片水印嵌入及恢复
1.计算过程
对载体图像的每个 32×32 分块，统计灰度值（0-255）出现的频率，按公式筛选出熵值最小的分块作为嵌入区域。

<img width="252" height="109" alt="图片1" src="https://github.com/user-attachments/assets/187cb4d5-79fd-41e1-acbd-9cd5f0c55cc6" />

2.水印嵌入计算
可见水印：直接用 32×32 水印像素值替换最小熵分块的像素值。
不可见水印：对最小熵分块中每个像素，根据水印与载体的灰度差动态调整强度（差越大调整幅度略大，限制在 2-6 之间），亮处加值、暗处减值，并记录修改位置（生成 0-1 掩码）。

3.水印提取计算
对比原始载体与含水印图像的最小熵分块，根据掩码找到修改过的像素：亮处像素差值减半还原亮部，暗处像素直接减差值还原暗部；再通过灰度裁剪（限制在原始水印灰度范围）和归一化，校准提取结果的灰度分布。

结果：

<img width="298" height="242" alt="图片2" src="https://github.com/user-attachments/assets/8a386327-9e9b-4cdf-a64a-95381fd1d028" />
<img width="251" height="256" alt="图片3" src="https://github.com/user-attachments/assets/c402af98-52f7-437e-8a75-bed8a22db9c5" />
<img width="238" height="220" alt="图片4" src="https://github.com/user-attachments/assets/8454d5fa-fd07-4448-9383-de813dd90f87" />
