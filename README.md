# yolov8_segmentation_ncnn

# How to build ubuntu 20.04
## 1. Git clone ncnn repo with submodule
```
cd yolov8_segmentation_ncnn
git clone https://github.com/Tencent/ncnn.git 
cd ncnn
git submodule update --init
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON ..
make -j$(nproc)
make install
```
[reference](https://github.com/Tencent/ncnn/wiki/how-to-build#raspberry-pi)

## 2. Add path ncnn to CMakeLists.txt

```
#NCNN
set(ncnn_DIR ./ncnn/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
```

## 3. Build project
```
cd yolov8_segmentation_ncnn
mkdir -p build && cd build
cmake ../
```

## 4. How to convert model
```
cd model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt
cd ../ultralytics
conda create --name yolov8-ncnn-seg-env python=3.8 -y
conda activate yolov8-ncnn-seg-env
pip install -r  requirements.txt
python3 convert_seg.py # create yolov8s-seg.onnx
../ncnn/build/install/bin/onnx2ncnn ../model/yolov8s-seg.onnx ../model/yolov8s-seg.param ../model/yolov8s-seg.bin --> config name model in main.cpp
```

## 5.Options
### 1. Git clone yolov8

```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
git checkout b9b0fd8bf409c822b7fcb21d65722b242f5307fc
conda create --name yolov8-ncnn-seg-env python=3.8 -y
conda activate yolov8-ncnn-seg-env
pip install -r  requirements.txt
```

### 1.1 Modify forward method of class C2f(nn.Module):

```
    def forward(self, x):
        # """Forward pass through C2f layer."""
        # y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, 1))
```
### 1.2 Modify forward method of class Detect(nn.Module)
```
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        return x_cat
        # if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
        #     box = x_cat[:, :self.reg_max * 4]
        #     cls = x_cat[:, self.reg_max * 4:]
        # else:
        #     box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        # if self.export and self.format in ('tflite', 'edgetpu'):
        #     # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
        #     # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
        #     # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
        #     img_h = shape[2] * self.stride[0]
        #     img_w = shape[3] * self.stride[0]
        #     img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
        #     dbox /= img_size

        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)
```

## 1.3 Modify forward method_of class Segment(Detect):
```
    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        # return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
        # !< https://github.com/FeiGeChuanShu/ncnn-android-yolov8
        return (torch.cat([x, mc], 1).permute(0, 2, 1), p.view(bs, self.nm, -1)) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))
```

## 6.Results

<div align="center">
<p>
<img src="image/seg.gif" width="600"/>
</p>