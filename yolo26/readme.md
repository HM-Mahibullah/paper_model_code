## create uv
```bash
uv venv   
source .venv/bin/activate 


cd ~/fps
source bin/activate

```

## pip upgrade and pytorch download
```bash
uv pip install pip   
pip install ultralytics==8.4.137 ncnn

```


## Transfer file  from local to rasp
```bash
scp /path/to/local/file.txt pi@<raspberry_pi_ip>:/home/pi/fps/
```



## Unzip file
```bash
cd ~/fps
python -m zipfile -e paper2_backbone_edge_bench.zip .
mv paper13_edge_bench/* .
rmdir paper13_edge_bench
ls

```

## removed file

cd ~/fps
rm -r bb_saconv_ncnn_model bench_images bb_saconv.pt saconv_backbone.py pi5_bench.py jetson_bench.py
ls





## running in rasp-5 

echo performance | sudo tee /sys/devices/system/cpu/cpu*/scaling_governor
python pi5_bench.py


## how to github large file 

