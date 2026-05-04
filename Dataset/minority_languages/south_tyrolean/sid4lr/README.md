# Slot and Intent Detection for Low Resource language varieties (SID4LR)

Welcome to the repository for the VarDial 2023 shared task on slot and intent detection for low resource language varieties. We follow the format of xSID ([van der goot et al, 2021](https://aclanthology.org/2021.naacl-main.197.pdf)). The data looks as follows:

```
# text: Cancel my reminder to text Dad
# intent: reminder/cancel_reminder
# slots: 10:18:reminder/noun,22:30:reminder/todo
1       Cancel  reminder/cancel_reminder        O
2       my      reminder/cancel_reminder        B-reference
3       reminder        reminder/cancel_reminder        O
4       to      reminder/cancel_reminder        O
5       text    reminder/cancel_reminder        B-reminder/todo
6       Dad     reminder/cancel_reminder        I-reminder/todo

```
This example sentence contains two slots ('my' and 'text dad'), and its intent is `reminder/cancel_reminder`.

The target languages are:

- South Tyrolean
- Neapolitan
- Swiss German

## Baseline
Besides the data, we also provide a baseline, which is the same baseline as in the original xSID paper (trained on the English data), with an updated version of MaChAmp ([van der goot et al, 2021](https://aclanthology.org/2021.eacl-demos.22.pdf)). This model uses an mBERT encoder, and a separate decoder head for each task, one for slot detection (with a CRF layer), and one for intent classification. Default parameters of MaChAmp are used. The baseline model can be downloaded from: https://itu.dk/people/robv/data/baseline-sid4lr.tar.gz . 

[![Baseline_overview](scripts/baseline.jpg)]()

The baseline was trained with the command (see also `scripts/baseline.sh`):

```
python3 train.py --dataset_configs ../scripts/sid4lr.json
```

It can be used for prediction (after unpacking in the MaChAmp root folder) using a command like: 

```
python3 predict.py sid4lr/2023.01.12_07.47.51/model.pt ../xSID-0.4/nap.valid.conll nap.valid-machamp.conll
```


## Contents
This repository contains:

- tgt_data: The target data for the shared task. In this folder, you will find the development datasplits, and the predictions of the baseline model.
- xSID-0.4: The training data for the shared task. You are allowed to use other data to train on, as long as it is not annotated for SID in the target languages. Note that the training data for all languages other than English is automatically translated.
- `scripts/baseline.sh`: the commands to run to reproduce the baseline used for the shared task. This is the baseline model from the original xSID paper, with an updated version of MaChAmp ([van der goot et al, 2021](https://aclanthology.org/2021.eacl-demos.22.pdf)).
- `scripts/sid4lr.json`: the baseline dataset configuration file, which is necessary to train the baseline model.
- `scripts/sidEval.py`: the official evaluation code. Reports span-f1 for slots and accuracy for intents. Additionally reports the % of completely correct utterances, and the loose and unlabeled span-f1 (see also [van der goot et al, 2021](https://aclanthology.org/2021.naacl-main.197.pdf))


The evaluation script can be used as follows:
```
python3 scripts/sidEval.py xSID-0.4/en.valid.conll machamp/logs/sid4lr/2023.01.12_07.47.51/SID4LR.out
recall:    0.9486754966887417
precision: 0.9597989949748744
slot-f1:   0.9542048293089093
intents:   1.0
fullCor:   0.9066666666666666

unlabeled
ul_recall:    0.9668874172185431
ul_precision: 0.9782244556113903
ul_slot-f1:   0.9725228975853455

loose (partial overlap with same label)
l_recall:    0.9817880794701986
l_precision: 0.9916247906197655
l_slot-f1:   0.9866819189166779
```

The baseline obtains the following scores on the target dev data:

|                | Intent-acc | Slot F1 |
|----------------|------------|---------|
| South Tyrolean | 0.6100     | 0.4461  |
| Swiss German   | 0.5167     | 0.2623  |
| Neapolitan     | 0.6100     | 0.4801  |

## Other data
A list of additional slot and intent detection datasets can be found on: https://github.com/yizhen20133868/Awesome-SLU-Survey (https://arxiv.org/pdf/2204.08582.pdf), another list can be found in the [MASSIVE paper](https://arxiv.org/pdf/2204.08582.pdf).


## Citation
The Swiss German and Neapolitan parts of the data have been introduced in:

```
@inproceedings{2023-findings-vardial,
  title = "Findings of the {V}ar{D}ial Evaluation Campaign 2023",
  author = {Aepli, No{\"e}mi and {\c{C}}{\"o}ltekin, {\c{C}}a{\u{g}}r{\i} and van der Goot, Rob and Jauhiainen, Tommi and Kazzaz, Mourhaf and Ljube{\v{s}}i{\'c}, Nikola and North, Kai and Plank, Barbara and Scherrer, Yves and Zampieri, Marcos},
  booktitle = "Proceedings of the Tenth Workshop on NLP for Similar Languages, Varieties and Dialects",
  month = may,
  year = "2023",
  address = "Dubrovnik, Croatia",
  publisher = "Association for Computational Linguistics",
}
```

The original xSID dataset (https://bitbucket.org/robvanderg/xsid) is from:
```
@inproceedings{van-der-goot-etal-2021-masked,
    title = "From Masked Language Modeling to Translation: Non-{E}nglish Auxiliary Tasks Improve Zero-shot Spoken Language Understanding",
    author = {van der Goot, Rob  and
      Sharaf, Ibrahim  and
      Imankulova, Aizhan  and
      {\"U}st{\"u}n, Ahmet  and
      Stepanovi{\'c}, Marija  and
      Ramponi, Alan  and
      Khairunnisa, Siti Oryza  and
      Komachi, Mamoru  and
      Plank, Barbara},
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.197",
    doi = "10.18653/v1/2021.naacl-main.197",
    pages = "2479--2497"
}
```

