import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# --------------------- 步骤1：读取图像 ---------------------
carrier_img = cv2.imread("C:/Users/25689/Desktop/task/main.jpg", 0)  
logo_img = cv2.imread("C:/Users/25689/Desktop/task/logo.png", 0)      

assert carrier_img.shape == (512, 512), "载体图像尺寸需为512×512"
assert logo_img.shape == (32, 32), "水印Logo尺寸需为32×32"

# 记录水印原图的灰度范围（用于后续校准）
logo_min = np.min(logo_img)
logo_max = np.max(logo_img)
print(f"原始水印灰度范围：{logo_min}~{logo_max}")

# --------------------- 步骤2：图像分块与信息熵计算 ---------------------
block_size = 32
min_entropy = float('inf')
min_block_pos = (0, 0)  

for i in range(0, 512, block_size):
    for j in range(0, 512, block_size):
        block = carrier_img[i:i+block_size, j:j+block_size]
        hist, _ = np.histogram(block, bins=256, range=(0, 256))
        prob = hist / (block_size * block_size)
        entropy = -np.sum([p * np.log2(p) for p in prob if p > 0])
        
        if entropy < min_entropy:
            min_entropy = entropy
            min_block_pos = (i, j)

print(f"最小信息熵: {min_entropy:.4f}")
print(f"最小熵分块位置: 左上角坐标({min_block_pos[0]}, {min_block_pos[1]})")

# --------------------- 步骤3：基础水印嵌入 ---------------------
watermarked_img = carrier_img.copy()
i, j = min_block_pos
watermarked_img[i:i+block_size, j:j+block_size] = logo_img
cv2.imwrite('watermarked_basic.jpg', watermarked_img)

# --------------------- 步骤4：不可见水印嵌入---------------------
watermarked_stealth = carrier_img.copy()
i, j = min_block_pos
block = watermarked_stealth[i:i+block_size, j:j+block_size]

# 嵌入规则：根据水印与载体的灰度差，按比例微调（强度与灰度差挂钩，避免过度调整）
modification_mask = np.zeros((32, 32), dtype=np.uint8)
base_strength = 2  # 基础强度
for x in range(32):
    for y in range(32):
        logo_gray = logo_img[x, y]
        carrier_gray = block[x, y]
        # 计算灰度差（归一化到0-1），差越大调整幅度越大（但限制最大强度）
        gray_diff = abs(logo_gray - carrier_gray) / 255  # 归一化差
        strength = int(base_strength * (1 + gray_diff * 2))  # 动态强度（2-6）
        strength = min(strength, 6)  # 最大强度不超过6
        
        if logo_gray > carrier_gray and carrier_gray + strength <= 255:
            watermarked_stealth[i+x, j+y] += strength
            modification_mask[x, y] = 1
        elif logo_gray < carrier_gray and carrier_gray - strength >= 0:
            watermarked_stealth[i+x, j+y] -= strength
            modification_mask[x, y] = 1

cv2.imwrite('watermarked_stealth.jpg', watermarked_stealth)

# --------------------- 步骤5：水印提取 ---------------------
extracted_logo = np.zeros((32, 32), dtype=np.uint8)
block_extract = watermarked_stealth[i:i+block_size, j:j+block_size]
for x in range(32):
    for y in range(32):
        carrier_gray = carrier_img[i+x, j+y]
        if modification_mask[x, y] == 1:
            # 调整提取逻辑：缩小亮度提升幅度，避免过亮
            if block_extract[x, y] > carrier_gray:
                # 水印偏亮时，仅提升载体灰度的1/2（原逻辑是+strength，容易过亮）
                delta = min(30, block_extract[x, y] - carrier_gray)  # 限制最大提升
                extracted_gray = carrier_gray + int(delta * 0.5)  # 缩小提升幅度
            else:
                # 水印偏暗时，正常降低灰度
                delta = min(30, carrier_gray - block_extract[x, y])
                extracted_gray = carrier_gray - delta
        else:
            extracted_gray = carrier_gray
        
        #将提取的灰度压缩到原始水印的灰度范围内
        extracted_gray = np.clip(extracted_gray, logo_min, logo_max)
        extracted_logo[x, y] = extracted_gray

extracted_logo = cv2.normalize(
    extracted_logo, None, 
    alpha=logo_min, beta=logo_max,  # 映射到原始水印的灰度范围
    norm_type=cv2.NORM_MINMAX
)
extracted_logo = extracted_logo.astype(np.uint8)

cv2.imwrite('extracted_logo.jpg', extracted_logo)

# --------------------- 显示结果 ---------------------
plt.figure(figsize=(12, 14))
plt.subplot(4, 2, 1)
plt.imshow(carrier_img, cmap='gray')
plt.title('载体图像')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.imshow(logo_img, cmap='gray')
plt.title('原始水印'.format(logo_min, logo_max))
plt.axis('off')

plt.subplot(4, 2, 3)
plt.imshow(watermarked_img, cmap='gray')
plt.title('基础水印嵌入结果')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(watermarked_stealth, cmap='gray')
plt.title('不可见水印嵌入结果')
plt.axis('off')

plt.subplot(4, 2, 5)
plt.imshow(extracted_logo, cmap='gray')
plt.title('优化后提取的水印')
plt.axis('off')


plt.tight_layout()
plt.show()