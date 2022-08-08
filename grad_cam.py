import cv2
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def processing_image(img_path, inputsize):
    # 讀取影像轉為輸入
    # 變換BGR通道至RGB
    x = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (inputsize, inputsize))
    # 加入batch
    x = np.expand_dims(x, axis=0)
    
    return x

def gradcam(model, x, last_layer_name):
    # 取得影像的分類類別
    preds = model.predict(x)

    pred_class = np.argmax(preds[0])
    
    # 預測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer(last_layer_name)
    
    # 求得分類的神經元對於最後一層 convolution layer 的梯度(last_conv_layer.output對pred_output做偏微分)
    grads = K.gradients(pred_output, last_conv_layer.output)[0] # 和last_conv_layer.output的維度相同

    
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads, axis=(0, 1, 2)) # 一維的特徵向量
    
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)

    return heatmap

def plot_heatmap(heatmap, img_path, pred_class_name):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    # 正規化
    heatmap /= np.max(heatmap)
    # 讀取影像
    img = cv2.imread(img_path)
    
    plt.figure()
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    plt.imshow(im)
    
    # 以 0.3 透明度繪製熱力圖
    plt.imshow(heatmap, cmap='jet', alpha=0.3)
    
    plt.title(pred_class_name)
    
    plt.show()

'''
model = loadmodel() # 將model導入

# print(model.summary())
img_path = './img/10.jpg' # 測試相片

img = processing_image(img_path, 224) # 相片預處理

heatmap = gradcam(model, img, pred_class_name) # 產生熱力圖

plot_heatmap(heatmap, img_path, pred_class_name) # 熱力圖視覺化
'''