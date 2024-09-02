# Assist Non-native Viewers: Multimodal Cross-Lingual Summarization for How2 Videos
The original conference version was accepted by *EMNLP 2022*, and the extended journal version has been accepted by *TPAMI*.

## Data Preparing
The reorganized How2-MCLS text data can be downloaded from here [[Baidu Netdisk, Passcode: 6cd9]](https://pan.baidu.com/s/1Kj2F6N4dC_1qZ89QYvR_EA), as well as video features [[Baidu Netdisk, Passcode: eqqj]](https://pan.baidu.com/s/1JxkceABDIDkkP3SS3ejX5g) (derived from the original How2 dataset). The original How2 dataset for multimodal summarization is provided by [https://github.com/srvk/how2-dataset](https://github.com/srvk/how2-dataset).

## Preprocessing
Some demo data is placed in "data/demo_data" folder, and you can replace the demo data with the full How2-MCLS dataset, following the format of "data/demo_data" folder. Then run the following command to preprocess the data.

 ```python
python preprocess.py #Please modify the data storage path configuration.
 ```

## Training and Prediction
You can run the following script commands to execute the training and prediction procedures of the proposed models, VDF, VDF-TS-E, and VDF-TS-V.

VDF

```
./run_scripts/VDF.sh
```

VDF-TS-E

```
./run_scripts/VDF-TS-E.sh
```

VDF-TS-V

```
./run_scripts/VDF-TS-V.sh
```

Alternatively, we also provide a well-trained first-stage model [[Baidu Netdisk, Passcode: rcqo]](https://pan.baidu.com/s/15AWUlc6I8kfwxSZ-MPrp0A) that you can choose to use directly to skip the first-stage training in the triple-stage training framework.

## Evaluation
[nmtpytorch](https://github.com/srvk/how2-dataset) library is used to evaluate models, which includes BLEU (1, 2, 3, 4), ROUGE-L, METEOR, and CIDEr evaluation metrics. 

As an alternative, [nlg-eval](https://github.com/Maluuba/nlg-eval) evaluation library can obtain the same evaluation scores as nmtpytorch.

In addition, [ROUGE](https://github.com/neural-dialogue-metrics/rouge) evaluation library is used to calculate the ROUGE (1, 2, L) score.

## Acknowledgement
We are very grateful that the code is based on [MFN](https://github.com/forkarinda/MFN), [nmtpytorch](https://github.com/srvk/how2-dataset), [fairseq](https://github.com/pytorch/fairseq), [machine-translation](https://github.com/tangbinh/machine-translation), and [Transformers](https://github.com/huggingface/transformers).

## Citation
```
@inproceedings{liu2022assist,
  title={Assist non-native viewers: Multimodal cross-lingual summarization for how2 videos},
  author={Liu, Nayu and Wei, Kaiwen and Sun, Xian and Yu, Hongfeng and Yao, Fanglong and Jin, Li and Zhi, Guo and Xu, Guangluan},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={6959--6969},
  year={2022}
}
@article{liu2024multimodal,
  title={Multimodal Cross-lingual Summarization for Videos: A Revisit in Knowledge Distillation Induced Triple-stage Training Method},
  author={Li, Mingjie and Huang, Po-Yao and Chang, Xiaojun and Hu, Junjie and Yang, Yi and Hauptmann, Alex},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  note = {Early Access},
  publisher={IEEE}
}
```
