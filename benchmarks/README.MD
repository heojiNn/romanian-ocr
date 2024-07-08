
# Experiment 1: Whole page - Printed Text
- Total words: 353
- Total characters: 2030

<!--![Balcescu](Balcescu.jpg)-->
<img src="benchmarks/balcesu_benchmark/Balcescu.jpg" alt="Balcescu" width="300"/>

## Evaluation Table:
| Method         | LDist    | Char Acc | Word Acc | Norm LD  | Lev Acc  | CER    | WER   |
|----------------|----------|----------|----------|----------|----------|--------|-------|
| bilstm-attn    | tba      | 0.98     | 0.83     | 0.05     | 0.94     | 0.06   | 0.12   |
| gpt4-tesseract | 54       | 0.97     | 0.89     | 0.0266   | 0.9734   | 0.0266 | 0.1108 |
| adobe          | 87       | 0.96     | 0.83     | 0.0427   | 0.9573   | 0.0428 | 0.1705 |
| easyocr        | 111      | 0.95     | 0.75     | 0.0547   | 0.9453   | 0.0547 | 0.2528 |
| astica         | 123      | 0.94     | 0.70     | 0.0605   | 0.9395   | 0.0606 | 0.3011 |
| gpt4o          | 169      | 0.92     | 0.62     | 0.0830   | 0.9170   | 0.0832 | 0.3807 |
| internvl       | 170      | 0.92     | 0.66     | 0.0837   | 0.9163   | 0.0837 | 0.3352 |
| tesseract      | 172      | 0.92     | 0.61     | 0.0845   | 0.9155   | 0.0847 | 0.3864 |
| mini-internvl  | 783      | 0.61     | 0.32     | 0.3855   | 0.6145   | 0.3855 | 0.6847 |
| minicpm        | 1184     | 0.42     | 0.14     | 0.5830   | 0.4170   | 0.5830 | 0.8608 |
| idefics2       | 1188     | 0.42     | 0.10     | 0.5849   | 0.4151   | 0.5849 | 0.9006 |
| trocr          | 2031     | 0.00     | 0.00     | 1.0000   | 0.0000   | 1.0000 | 1.0000 |
| pymupdf        | 2076     | -0.02    | -0.14    | 0.5081   | 0.4919   | 1.0222 | 1.1420 |

Training Evaluation

Best:
LD: 0.25, Char Acc: 0.98, Word Acc: 0.83, Lev Acc: 0.94, CER: 0.06, WER: 0.12

