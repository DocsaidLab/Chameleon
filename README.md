# Chameleon

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/Chameleon/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/Chameleon?color=ffa"></a>
    <a href="https://pypi.org/project/chameleon_docsaid/"><img src="https://img.shields.io/pypi/v/chameleon_docsaid.svg"></a>
    <a href="https://pypi.org/project/chameleon_docsaid/"><img src="https://img.shields.io/pypi/dm/chameleon_docsaid?color=9cf"></a>
</p>

![title](https://raw.githubusercontent.com/DocsaidLab/Chameleon/refs/heads/main/docs/title.webp)

Chameleon 是一套以 **PyTorch** 為基礎的深度學習工具集，提供了從神經網路構建到模型效能量測的完整模組化功能。專案中的元件都以 Registry 機制統一管理，使模型搭建與擴充更加彈性。

## 功能特色

- **模組化 Registry**：透過 `Registry` 對元件進行註冊與管理，可快速建立 backbone、neck、optimizers 等各式模組。
- **多樣化模型元件**：內建常用的卷積區塊、激活函式、正規化、池化等基礎元件，以及 ASPP、FPN、BiFPN 等進階結構。
- **計算資源評估工具**：提供 `calculate_flops` 函式，能輕鬆計算模型的 FLOPs、MACs 與參數量。
- **客製化指標**：內建 `NormalizedLevenshteinSimilarity` 等評估指標，方便進行文字相關任務的相似度量測。
- **實用工具**：包含模型權重初始化、模組替換、CPU 資訊取得與模型複雜度分析等輔助函式。

## 安裝

可直接使用 pip 從 PyPI 取得最新版套件：

```bash
pip install chameleon_docsaid
```

或是從原始碼安裝：

```bash
git clone https://github.com/DocsaidLab/Chameleon.git
cd Chameleon
pip install -e .
```

安裝時請確認 Python 版本為 3.10 以上，並已安裝 PyTorch 1.13 以上版本。

## 快速開始

以下示範如何利用 Registry 建立模型並計算 FLOPs：

```python
from chameleon import BACKBONES, build_neck
from chameleon.tools import calculate_flops

# 建立 backbone 與 neck
backbone = BACKBONES.build({'name': 'timm_resnet50'})
neck = build_neck('FPN', in_channels_list=[256, 512, 1024, 2048], out_channels=256)

# 計算模型 FLOPs
flops, macs, params = calculate_flops(backbone, (1, 3, 224, 224))
print(flops, macs, params)
```

## 專案結構簡介

- `chameleon/base`：基礎模組與工具函式，如 `PowerModule`、權重初始化等。
- `chameleon/modules`：常用的神經網路結構，包含 backbones 與 necks。
- `chameleon/metrics`：評估指標實作。
- `chameleon/registry`：模組註冊與建構的核心機制。
- `chameleon/tools`：計算 FLOPs 等實用工具。
- `docs`：詳細文件與教學。
- `tests`：單元測試，確保各模組功能正確。

更多使用範例與 API 說明請參閱 [`docs/chameleon.md`](docs/chameleon.md)。

## 貢獻

歡迎提出 issue 或 pull request 協助改進 Chameleon。提交程式碼前請先執行單元測試，確保變更不會造成既有功能的問題。

## 授權

本專案採用 [Apache License 2.0](LICENSE) 授權釋出。
