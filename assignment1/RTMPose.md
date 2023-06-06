# Assignment1

## **题目**：基于RTMPose的耳朵穴位关键点检测

**背景**：根据中医的“倒置胎儿”学说，耳朵的穴位反映了人体全身脏器的健康，耳穴按摩可以缓解失眠多梦、内分泌失调等疾病。耳朵面积较小，但穴位密集，涉及耳舟、耳轮、三角窝、耳甲艇、对耳轮等三维轮廓，普通人难以精准定位耳朵穴位。

**任务** 
1. Labelme标注关键点检测数据集（子豪兄已经帮你完成了） 
2. 划分训练集和测试集（子豪兄已经帮你完成了） 
3. Labelme标注转MS COCO格式（子豪兄已经帮你完成了） 
4. 使用MMDetection算法库，训练RTMDet耳朵目标检测算法，提交测试集评估指标 
5. 使用MMPose算法库，训练RTMPose耳朵关键点检测算法，提交测试集评估指标 
6. 用自己耳朵的图像预测，将预测结果发到群里 7.用自己耳朵的视频预测，将预测结果发到群里 需提交的测试集评估指标（不能低于baseline指标的50%）

- 目标检测Baseline模型（RTMDet-tiny）  

训练：
```python
python tools/train.py data/rtmdet_tiny_ear.py
```
测试：
```python
python tools/test.py data/rtmdet_tiny_ear.py work_dirs/rtmdet_tiny_ear/best_coco_bbox_mAP_epoch_165.pth 
```

```shell
DONE (t=0.02s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.813
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.967
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.843
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.843
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.843
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.843
06/06 23:08:14 - mmengine - INFO - bbox_mAP_copypaste: 0.813 0.967 0.967 -1.000 -1.000 0.813
06/06 23:08:15 - mmengine - INFO - Epoch(test) [11/11]    coco/bbox_mAP: 0.8130  coco/bbox_mAP_50: 0.9670  coco/bbox_mAP_75: 0.9670  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8130  data_time: 1.3234  time: 1.8108 
```

- 关键点检测Baseline模型（RTMPose-s）  

训练：
```python
python tools/train.py data/rtmpose-s-ear.py
```
测试：
```python
python tools/test.py data/rtmpose-s-ear.py work_dirs/rtmpose-s-ear/best_PCK_epoch_270.pth 
```
```shell
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.745
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.970
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.745
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.786
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.786
06/07 01:04:13 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
06/07 01:04:13 - mmengine - INFO - Evaluating AUC...
06/07 01:04:13 - mmengine - INFO - Evaluating NME...
06/07 01:04:13 - mmengine - INFO - Epoch(test) [6/6]    coco/AP: 0.745408  coco/AP .5: 1.000000  coco/AP .75: 0.970297  coco/AP (M): -1.000000  coco/AP (L): 0.745408  coco/AR: 0.785714  coco/AR .5: 1.000000  coco/AR .75: 0.976190  coco/AR (M): -1.000000  coco/AR (L): 0.785714  PCK: 0.971655  AUC: 0.123980  NME: 0.040596  data_time: 1.754922  time: 2.324906
```

**预测** 

```python
    
!python demo/topdown_demo_with_mmdet.py \
        data/rtmdet_tiny_ear.py \
        work_dirs/rtmpose-s-ear/best_coco_bbox_mAP_epoch_165.pth \
        data/rtmpose-s-ear.py \
        work_dirs/rtmpose-s-ear/best_PCK_epoch_270.pth \
        --input data/test/1.jpg \
        --output-root outputs/ear-RTMPose \
        --device cuda:0 \
        --bbox-thr 0.5 \
        --kpt-thr 0.5 \
        --nms-thr 0.3 \
        --radius 8 \
        --thickness 30 \
        --draw-bbox \
        --draw-heatmap \
        --show-kpt-idx
```


