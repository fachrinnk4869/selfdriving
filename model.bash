# install yolopv2 model
# wget https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
# mv yolopv2.pt ./src/yolop/src/yolop/data/weights

# install polylannet model
gdown https://drive.google.com/uc?id=1wSsAGa63ebo9GyCCS_6hdDR_c7QQsr5F
mkdir -p ./src/polylannet/src/experiments/exp/models
mv model_2695.pt ./src/polylannet/src/experiments/exp/models