# study
# trt_ppocr3
tensorrt ocr ppocr3  dynamic_shapes


env: 

ubuntu18.04 + cuda11.1 + tensorrt7.2.2.3

torch==1.8.1

##pth -> wts 

python gen_det_wts.py

python gen_rec_wts.py

## wts -> engine

cd convert/rec/

makedir build && cd build && cmake .. &&  make -j8

./rec

cd convert/det/

makedir build && cd build && cmake .. &&  make -j8

./det

## inference 

python predict_rec_trt.py

python predict_det_trt.py

