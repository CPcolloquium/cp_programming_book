# R/Pythonでできる計算論的精神医学

## 正誤表
[第１刷　正誤表](errata.md)



## ３章 生物物理学的モデル
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/3_Biophysical-model/3_Biophysical-model.ipynb)


## ４章 ニューラルネットワークモデル
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/4_Neural-network-model/4_Neural-network-model.ipynb)

## ５章 強化学習モデル
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/5_Reinforcement-learing-model.ipynb)

### 逆転学習課題
- [逆転学習課題のデモ](https://cba-toolbox.github.io/probabilistic-reversal-learning-task/demo_probabilistic-reversal-learning.html) : jsPsychで作成した逆転学習課題のデモです。このデモでは40試行で逆転があり、全80試行です。最後まで取り組むとcsvファイルがダウンロードされますので、パラメータ推定に使えます。

- [逆転学習課題のCBATリポジトリ](https://github.com/cba-toolbox/probabilistic-reversal-learning-task)

- 「逆転学習課題のデモ」データでパラメータ推定：　上記の「逆転学習課題のデモ」に取り組むと、最後にcsvファイルがダウンロードされます。このcsvファイルを使ってご自身のデータからパラメータ推定もできます。それにあたって、ダウンロードされたファイルを解析可能なフォーマットにするPRLpreprocessパッケージを作成しました。PRLpreprocessパッケージは、cmot4rパッケージ下にありますので、cmot4rパッケージをインストールすると入ります。以下のように、cmot4rパッケージをインストールしてから、` PRLpreprocess::PRL_preprocess_csv()`関数でデータの読み込み＆前処理を実行ください。あとは、本書で解説されている通りに作業をしていくとパラメータ推定ができます。なお、ご自身のR環境にremotesパッケージが入ってなければ`install.packages("remotes")`を実行してから利用ください。cmot4rのリポジトリは、[cba-toolbox/cmot4r](https://github.com/cba-toolbox/cmot4r)です。

``` r
# install.packages("remotes")
remotes::install_github("cba-toolbox/cmot4r")
data <- PRLpreprocess::PRL_preprocess_csv("data.csv")
```

## ６章 ベイズ推論モデル
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/6_Bayesian-inference-model.ipynb)

### 釣り人課題
- [釣り人課題のデモ](https://cba-toolbox.github.io/fisherman-task/demo_fisherman-task.html) : ６章ではビーズ課題について説明しましたが、釣り人課題は全く同じ構造の課題です。釣り人課題は、jsPsychで作成しました。最後まで取り組むとcsvファイルがダウンロードされますので、パラメータ推定に使えます。なお、釣り人課題のCBATリポジトリは、[こちら](https://github.com/cba-toolbox/fisherman-task)です。

- 釣り人課題のデータ読み込みと前処理用Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/6_Bayesian-inference-model/preprocess_fisherman_task_data.ipynb)：上の「釣り人課題のデモ」に取り組んだ際に出力されるcsvファイルを読み込んで前処理するためのColabです。


### 変動性のある逆転学習課題
- [変動性のある逆転学習課題のデモ](https://cba-toolbox.github.io/volatile-probabilistic-reversal-learning-task/demo_volatile-probabilistic-reversal-learning.html) : jsPsychで作成した変動性のある逆転学習課題のデモです。最後まで取り組むとcsvファイルがダウンロードされますので、パラメータ推定に使えます。なお、変動性のある逆転学習課題のCBATリポジトリは、[こちら](https://github.com/cba-toolbox/volatile-probabilistic-reversal-learning-task)

- 変動性のある逆転学習課題のデータ読み込みと前処理用のColab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CPcolloquium/cp_programming_book/blob/main/6_Bayesian-inference-model/preprocess_volatile_probabilistic_reversal_learning_task_data.ipynb)：上の「変動性のある逆転学習課題のデモ」に取り組んだ際に出力されるcsvファイルを読み込んで前処理するためのColabです。