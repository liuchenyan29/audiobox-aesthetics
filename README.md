# audiobox-aesthetics

Unified automatic quality assessment for speech, music, and sound.

[](Paper)

## Installation

This repository requires Python 3.9 and Pytorch 2.2 or greater. To install, you can clone this repo and run:
```
pip install -e .
```

## Pre-trained Models

Model | Link
|---|---|
All axes | [checkpoint.pt](https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt)

## Usage

How to run prediction:

1. Create a jsonl files with the following format
```
{"path":"/path/to/a.wav"}
{"path":"/path/to/b.wav"}
...
{"path":"/path/to/z.wav"}
```
or if you only want to predict aesthetic score from certain timestamp
```
{"path":"/path/to/a.wav", "start_time":0, "end_time": 5}
{"path":"/path/to/b.wav", "start_time":3, "end_time": 10}
```
and save it as `input.jsonl`

2. Run following command
```
audio-aes input.jsonl --ckpt "/path/to/checkpoint.pt" > output.jsonl
```

3. Output file will contains same number of rows as `input.jsonl`. Each rows contains 4 axes prediction with JSON-formatted dictionary. Check following table for more info:
    Axes name | Full name
    |---|---|
    CE | Content Enjoyment
    CU | Content Usefulness
    PC | Production Complexity
    PQ | Production Quality
    
    Output line example:
    ```
    {"CE": 5.146, "CU": 5.779, "PC": 2.148, "PQ": 7.220}
    ```



4. (Extra) If you want to extract only one axis (i.e. CE), post-process the output file with following command using `jq` utility: 
    
    ```jq '.CE' output.jsonl > output-aes_ce.txt```



## Evaluation dataset
We released our evaluation dataset consisted of 4 axes of aesthetic annotation scores. 

Here, we show an example on how to read and re-map each annotation to the actual audio file.
```
{
    "data_path": "/your_path/LibriTTS/train-clean-100/1363/139304/1363_139304_000011_000000.wav", 
    "Production_Quality": [8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 8.0, 5.0, 8.0, 8.0], 
    "Production_Complexity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
    "Content_Enjoyment": [8.0, 6.0, 8.0, 5.0, 8.0, 8.0, 8.0, 6.0, 8.0, 6.0], 
    "Content_Usefulness": [8.0, 6.0, 8.0, 7.0, 8.0, 9.0, 8.0, 6.0, 10.0, 7.0]
}
```
1. Recognize the dataset name from data_path. In the example, it is LibriTTS.
2. Replace "/your_path/" into your downloaded LibriTTS directory. 
3. Each axes contains 10 scores annotated by 10 different human annotators.

data_path | URL
|---|---|
LibriTTS |  https://openslr.org/60/
cv-corpus-13.0-2023-03-09 | https://commonvoice.mozilla.org/en/datasets
EARS | https://sp-uhh.github.io/ears_dataset/
MUSDB18 | https://sigsep.github.io/datasets/musdb.html
musiccaps | https://www.kaggle.com/datasets/googleai/musiccaps
(audioset) unbalanced_train_segments | https://research.google.com/audioset/dataset/index.html 
PAM | https://zenodo.org/records/10737388

## License
The majority of audiobox-aesthetics is licensed under CC-BY 4.0, as found in the LICENSE file.
However, portions of the project are available under separate license terms: [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm) is licensed under MIT license.

## Citation
If you found this repository useful, please use the following BibTeX entry.

```
@article{tjandra2025aes,
    title={Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound},
    author={Tjandra, Andros and Wu, Yi-Chiao and Guo, Baishan and Hoffman, John and Ellis, Brian and Vyas, Apoorv and Shi, Bowen and Chen, Sanyuan and Le, Matt and Zacharov, Nick and Wood, Carleigh and Lee, Ann and Hsu, Wei-ning},
    publisher={Meta AI},
    year={2025},
    url={https://ai.meta.com/research/publications/meta-audiobox-aesthetics-unified-automatic-quality-assessment-for-speech-music-and-sound/}
}
```

## Acknowledgements

Part of model code are copied from [https://github.com/microsoft/unilm/tree/master/wavlm](WavLM).

