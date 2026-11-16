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

## All script

```bash
cat > ~/fps/bench_paper.sh << 'EOF'
#!/bin/bash
set -e
cd ~/fps
ZIP="$1"
# clean previous paper (keeps venv: bin lib lib64 share pyvenv.cfg CACHEDIR.TAG)
rm -rf bb_* *_backbone.py pi5_bench.py jetson_bench.py bench_images runs weights *.yaml *_edge_bench
# extract and flatten
~/fps/bin/python -m zipfile -e "$ZIP" .
mv *_edge_bench/* .
rmdir *_edge_bench
rm -rf runs weights *.yaml "$ZIP"
ls
# max CPU speed
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
# benchmark + save result
echo "===== $ZIP =====" >> results.txt
~/fps/bin/python pi5_bench.py | tee -a results.txt
EOF
chmod +x ~/fps/bench_paper.sh
```

## then paper one by one 
```bash
scp "D:\rasp\yolo26\paper_code_backbone\paper-8-backbone\paper8_backbone_all_results.zip" pi@192.168.0.116:/home/pi/fps/

~/fps/bench_paper.sh paper8_backbone_all_results.zip

```


# Neck script


```bash
cat > ~/fps/bench_paper.sh << 'EOF'
#!/bin/bash
set -e
cd ~/fps
ZIP="$1"
[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }

# clean the previous paper (keeps the venv: bin lib lib64 share include pyvenv.cfg CACHEDIR.TAG, and results.txt)
rm -rf bb_* nk_* hd_* *.pt *_ncnn_model *_backbone.py *_neck.py *_head.py fix_eca_ncnn.py \
       pi5_bench.py jetson_bench.py bench_images runs weights *.yaml *_edge_bench

# extract and flatten (the edge folder holds the NCNN models, module files and bench script)
~/fps/bin/python -m zipfile -e "$ZIP" .
cp -rf *_edge_bench/* .
rm -rf *_edge_bench runs weights *.yaml "$ZIP"
ls

# max CPU speed
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null

# benchmark + save result
echo "===== $ZIP =====" >> results.txt
~/fps/bin/python pi5_bench.py | tee -a results.txt
EOF
chmod +x ~/fps/bench_paper.sh


```

# to move files all files

scp /media/Data/rasp/yolo26/paper_code_neck/*/*_neck_all_results.zip pi@192.168.0.116:/home/pi/fps/



cd ~/fps
for z in paper*_neck_all_results.zip; do ~/fps/bench_paper.sh "$z"; done



## for paer 18
~/fps/bench_paper.sh paper19_neck_all_results.zip
