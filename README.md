# TTS project

## Installation guide

Clone repository 
```shell
git clone https://github.com/marina-shesha/TTS.git
```
Download requirements
```shell
pip install -r TTS/requirements.txt
```
Download data
```shell
%%bash 
#install libraries
pip install gdown==4.5.4 --no-cache-dir

#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

gdown https://drive.google.com/u/0/uc?id=1m4kdtbVBrvxbYSzl1xZ7xWU_aUvoFe8l&export=download
mv train.txt data/

#download Waveglow
gdown https://drive.google.com/u/0/uc?id=1ktzlAa5a4ilzPf6mU8Mdt3eSYPOGrlYx&export=download
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

gdown https://drive.google.com/u/0/uc?id=1-MbvRyUo4OACk_sQhmG9-aIE6XJ3NxSh&export=download
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null

# we will use waveglow code, data and audio preprocessing from this repo
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/audio .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .
```
We might have to install the following code so that cuda doesn't fail.
```shell
!pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Training

Run train.py with base_config.json

```shell
%run -i TTS/train.py --config TTS/hw_tts/configs/base_config.json
```
## Test 

Download checkpount and config

```shell
gdown https://drive.google.com/u/0/uc?id=1-GBrqxyL3QHcDuzs_pJJnr6rNveVpKsC&export=download
gdown https://drive.google.com/u/0/uc?id=1-xnNtmOhmo0wz7VXQm4NBj8r7fuByz5s&export=download
```

Run test.py file to get audio preductions

```shell
%run -i TTS/test.py \
--resume /content/checkpoint-epoch300.pth\
--config /content/TTS/hw_tts/configs/base_config.json\
-o test_clean_out.json
```
Audio preductions will be located in dir "test_results". Use  

```shell
from IPython import display
display.display(display.Audio(audio_path))
```
for visualization audio.


You can see usage of this this instruction in ```fastspeech2.ipynb```
