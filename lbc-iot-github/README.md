# LBC-IoT 复现实验

在上一级工作区的 PowerShell 运行以下命令，使用现有虚拟环境。

```powershell
.\.venv\Scripts\python.exe -u LBC-IoT/lbc_runtime.py
.\.venv\Scripts\python.exe -u LBC-IoT/train.py --preset smoke
.\.venv\Scripts\python.exe -u LBC-IoT/recover.py --pairs 16 --trials 1
```

第一条检查加解密和数据生成，第二条用小数据训练五轮区分器并在新数据上测试，第三条枚举六轮最后一轮的全部 65536 个子密钥。恢复脚本默认读取最近训练完成的模型，也可用 `--run` 指定模型所在目录。训练记录和模型保存在 `runs`，排名和完整候选分数保存在 `results`。

训练沿用作者的网络和样本生成方法。负样本给同一对密文异或同一个随机掩码，所以两条密文的异或差分保持不变。小规模配置用于验证代码。要按作者训练 Notebook 的八轮配置运行，可使用 `train.py --preset author`，默认 100 万训练样本、10 个残差块、40 个训练周期。脚本按模型轮数加一安排子密钥排名。

作者的排名 Notebook 中，训练和加载示例的轮数不一致。本地脚本从训练记录读取轮数和差分。排名采用作者的计分方式，即统计每个候选对应的网络得分超过 0.5 的样本数。分数相同时报告并列排名区间。

运行前，从[作者仓库提交 4f7b718](https://github.com/CryptoUSM/deep-lbc-slim/tree/4f7b7185c1a46e5e07665eaeb81d3028381c9143) 取得 `LBC-IoT_Github` 目录，放到本项目的 `upstream` 下。脚本会从那里读取 `lbciot.py` 和 `train_nets.py`。上面的命令假设你把项目目录命名为 `LBC-IoT`，并在其上一级目录配置了带 TensorFlow、Keras 和 NumPy 的 `.venv`。[论文原文入口](https://doi.org/10.1016/j.jisa.2023.103531)
