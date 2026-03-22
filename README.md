# 🤖 DRL Portfolio: Tối Ưu Danh Mục Đầu Tư Bằng Deep Reinforcement Learning
File dữ liệu: https://drive.google.com/drive/folders/1OvhywCGDasegz_Y1o3garUkXt-PHog9z
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)
![Market](https://img.shields.io/badge/Market-VN100-red)

**Hệ thống phân bổ danh mục chứng khoán Việt Nam kết hợp Recurrent PPO-LSTM, PhoBERT Sentiment và kiểm soát rủi ro CVaR**

*Hợp tác nghiên cứu với Chứng khoán Rồng Việt · GPA 3.4/4 · AI in Quant Trading: 4.0/4*

</div>

---

## 📊 Kết Quả Nổi Bật

| Chỉ số | **Mô hình** | Buy & Hold | VN-Index |
|--------|:-----------:|:----------:|:--------:|
| Lợi nhuận tích lũy (2022–2025) | **~89.34%** | ~16.78% | ~47.86% |
| Sharpe Ratio | **0.716** | 0.288 | 0.516 |
| Sortino Ratio | **1.034** | 0.334 | 0.677 |
| Max Drawdown | ✅ Kiểm soát CVaR | — | — |

---

## 🗺️ Kiến Trúc Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NGUỒN DỮ LIỆU THÔ                          │
└──────────────────────┬──────────────────────┬───────────────────────┘
                       │                      │
           ┌───────────▼──────────┐  ┌────────▼──────────────┐
           │  📈 Dữ liệu giá TCBS │  │  📰 Crawl tin tức     │
           │  (VN100, 2014–2025)  │  │  (CafeF, VietStock,   │
           │                      │  │   TinNhanhCK)         │
           └───────────┬──────────┘  └────────┬──────────────┘
                       │                      │
     ┌─────────────────▼──────────┐  ┌────────▼──────────────────┐
     │  1️⃣  Lọc cổ phiếu &       │  │  2️⃣  Crawl news of stocks  │
     │  Tính chỉ báo kỹ thuật     │  │  (crawl_news_of_stocks     │
     │  (loc_co_phieu_chi_bao     │  │   .ipynb)                 │
     │   .ipynb)                  │  └────────┬──────────────────┘
     │                            │           │
     │  OUTPUT:                   │  ┌────────▼──────────────────┐
     │  • top_30_stocks_data.xlsx │  │  3️⃣  Huấn luyện PhoBERT   │
     │  • trainvn_risk.xlsx       │  │  (phobert_train.ipynb)    │
     │  • testvn_risk.xlsx        │  │                           │
     └─────────────────┬──────────┘  │  OUTPUT:                  │
                       │             │  • best_phobert_multitask  │
                       │             │    .pt                    │
                       │             └────────┬──────────────────┘
                       │                      │
                       │             ┌────────▼──────────────────┐
                       │             │  4️⃣  Chấm điểm tin tức    │
                       │             │  (cham_diem_tin_tuc        │
                       │             │   .ipynb)                 │
                       │             │                           │
                       │             │  OUTPUT:                  │
                       │             │  • cafef_end(2014-2024)   │
                       │             │    .xlsx (có sentiment)   │
                       │             │  • tinnhanhchungkhoan_end │
                       │             │    (2014-2024).xlsx       │
                       │             └────────┬──────────────────┘
                       │                      │
              ┌────────▼──────────────────────▼──────────┐
              │              MERGE DỮ LIỆU               │
              │  Giá + Chỉ báo kỹ thuật + Embedding PhoBERT│
              │                                           │
              │  OUTPUT:                                  │
              │  • train2025_sentiment_with_emb.csv       │
              │  • test2025_sentiment_with_emb2.csv       │
              └────────────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │  5️⃣  Train & Backtest Recurrent PPO-LSTM   │
              │  (recurrent_ppo_lstm.ipynb)               │
              │                                           │
              │  Ensemble: Dynamic Multi-Seed (30 models) │
              │  Reward: CVaR-penalized multi-objective   │
              │                                           │
              │  OUTPUT:                                  │
              │  • trained models (.zip)                  │
              │  • allocation_*.csv                       │
              │  • history_*.csv                          │
              └────────────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼──────────────────────┐
              │  6️⃣  Kiểm định mô hình DRL               │
              │  (kiem_dinh_drl.ipynb)                    │
              │                                           │
              │  • Kiểm định xu hướng (CAPM + LASSO)     │
              │  • Kiểm định tương lai (IG Alignment)    │
              │  • Phân tích chế độ thị trường           │
              │                                           │
              │  INPUT:                                   │
              │  • test2025_risk.xlsx                     │
              │  • ig_explanation_*.csv                   │
              │  • beta_single_step.csv                   │
              │  • beta_multi_step_W5/W20/W60.csv        │
              │  • vnm.csv                                │
              │  • allocation_*.csv                       │
              └───────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục

```
DRL-Portfolio/
│
├── 📓 notebooks/
│   ├── 01_loc_co_phieu_chi_bao.ipynb        # Lọc cổ phiếu & tính chỉ báo kỹ thuật
│   ├── 02_crawl_news_of_stocks.ipynb        # Thu thập tin tức từ CafeF, VietStock, TNCK
│   ├── 03_phobert_train.ipynb               # Huấn luyện PhoBERT multi-task
│   ├── 04_cham_diem_tin_tuc.ipynb           # Chấm điểm sentiment & tạo embedding
│   ├── 05_recurrent_ppo_lstm.ipynb          # Train & backtest Recurrent PPO
│   └── 06_kiem_dinh_drl.ipynb               # Kiểm định thống kê mô hình
│
├── 📂 data/
│   ├── raw/
│   │   ├── du_lieu_tcbs_vn.xlsx             # Dữ liệu giá thô từ TCBS (VN100, 2014–2025)
│   │   └── Macp.xlsx                        # Danh sách mã cổ phiếu
│   │
│   ├── processed/
│   │   ├── top_30_stocks_data.xlsx          # 30 cổ phiếu được lọc sau backtesting MA
│   │   ├── trainvn_risk.xlsx                # Dữ liệu train với chỉ báo kỹ thuật
│   │   ├── testvn_risk.xlsx                 # Dữ liệu test với chỉ báo kỹ thuật
│   │   ├── cafef_end(2014-2024).xlsx        # Tin tức CafeF sau khi chấm sentiment
│   │   └── tinnhanhchungkhoan_end(2014-2024).xlsx  # Tin tức TNCK sau khi chấm sentiment
│   │
│   └── embeddings/
│       ├── train2025_sentiment_with_emb.csv # Train: giá + chỉ báo + PhoBERT embedding
│       └── test2025_sentiment_with_emb2.csv # Test: giá + chỉ báo + PhoBERT embedding
│
├── 🤖 models/
│   ├── best_phobert_multitask.pt            # Checkpoint PhoBERT tốt nhất
│   └── trained_seeds/                       # 30 model seeds từ Dynamic Multi-Seed Ensemble
│
├── 📊 results/
│   ├── allocation_*.csv                     # Phân bổ danh mục theo từng phương pháp
│   ├── history_*.csv                        # Lịch sử giao dịch
│   ├── ig_explanation_*.csv                 # Integrated Gradients explanations
│   ├── beta_single_step.csv                 # Beta kiểm định 1-day
│   ├── beta_multi_step_W5.csv               # Beta kiểm định 5-day
│   ├── beta_multi_step_W20.csv              # Beta kiểm định 20-day
│   ├── beta_multi_step_W60.csv              # Beta kiểm định 60-day
│   └── vnm.csv                              # Dữ liệu VNM ETF (proxy thị trường)
│
├── 📋 train_phobert_multitask.csv           # Tập train PhoBERT (labeled sentiment)
├── 📋 test_phobert_multitask.csv            # Tập test PhoBERT
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng Dẫn Chạy Theo Luồng

### Bước 1 & 2 (Song song): Chuẩn bị dữ liệu

**1️⃣ Lọc cổ phiếu & tính chỉ báo kỹ thuật**

Mở `notebooks/01_loc_co_phieu_chi_bao.ipynb`

```python
# Sửa đường dẫn input ở đầu notebook:
INPUT_XLSX = "data/raw/du_lieu_tcbs_vn.xlsx"   # ← dữ liệu giá thô từ TCBS

# Đường dẫn output (tự động tạo):
OUT_TRAIN  = "data/processed/trainvn_risk.xlsx"
OUT_TEST   = "data/processed/testvn_risk.xlsx"
```

Notebook này thực hiện:
- Backtesting chiến lược MA Cross trên toàn bộ VN100 để lọc top 30 cổ phiếu
- Tính 8 chỉ báo kỹ thuật: RSI-30, MACD, Bollinger Bands, CCI-30, ADX-30, SMA-30, SMA-60
- Chia train/test theo thời gian, xuất `trainvn_risk.xlsx` và `testvn_risk.xlsx`

---

**2️⃣ Thu thập tin tức từ các nguồn**

Mở `notebooks/02_crawl_news_of_stocks.ipynb`

```python
# Crawl từ 3 nguồn song song với bước 1:
# - Tin Nhanh Chứng Khoán (sitemap-based)
# - VietStock (Selenium + ngày tháng)
# - CafeF (Selenium, theo từng mã)

# File trung gian (tự xuất sau mỗi nguồn):
# data/raw/tinnhanhck_raw.xlsx
# data/raw/vietstock_raw.xlsx
# data/raw/cafef_raw.xlsx

# Sau khi crawl xong, chạy cell "Crawl pHead" để lấy nội dung đầy đủ:
excel_path = "data/raw/vietstock_raw.xlsx"   # ← đổi theo nguồn đang xử lý

# Output cuối cùng:
out_xlsx = "data/raw/output_with_content.xlsx"
```

---

### Bước 3: Huấn luyện PhoBERT

Mở `notebooks/03_phobert_train.ipynb`

```python
# ====================== CẤU HÌNH ======================
TRAIN_CSV  = "train_phobert_multitask.csv"   # ← file labeled sentiment
TEST_CSV   = "test_phobert_multitask.csv"    # ← file test labeled

TICKER_FILES = [
    "data/raw/Macp.xlsx",                    # ← danh sách mã cổ phiếu để entity masking
]

MODEL_NAME = "vinai/phobert-large"           # auto-download từ HuggingFace
# Lưu checkpoint tốt nhất:
# → models/best_phobert_multitask.pt
```

> ⚠️ **Yêu cầu GPU.** Chạy trên Google Colab (A100) hoặc Kaggle (P100). CPU sẽ rất chậm.

---

### Bước 4: Chấm điểm tin tức & tạo embedding

Mở `notebooks/04_cham_diem_tin_tuc.ipynb`

```python
# ====================== CẤU HÌNH ======================
CKPT_PATH = "models/best_phobert_multitask.pt"   # ← checkpoint từ bước 3

INPUT_FILES = [
    "data/raw/cafef_end(2014-2024).xlsx",                  # ← tin tức CafeF đã crawl
    "data/raw/tinnhanhchungkhoan_end(2014-2024).xlsx"      # ← tin tức TNCK đã crawl
]

TICKER_FILES = [
    "data/raw/Macp.xlsx",                                  # ← danh sách mã để entity masking
]

# Output: file gốc được ghi thêm cột sentiment_score + embedding (1024 chiều)
# → data/processed/cafef_end(2014-2024).xlsx       (có sentiment)
# → data/processed/tinnhanhchungkhoan_end(2014-2024).xlsx (có sentiment)
```

Sau khi chấm điểm xong, **merge thủ công** embedding vào dữ liệu giá:

```
data/processed/trainvn_risk.xlsx  + embedding → data/embeddings/train2025_sentiment_with_emb.csv
data/processed/testvn_risk.xlsx   + embedding → data/embeddings/test2025_sentiment_with_emb2.csv
```

---

### Bước 5: Huấn luyện & Backtest Recurrent PPO

Mở `notebooks/05_recurrent_ppo_lstm.ipynb`

```python
# ====================== TRAINING CONFIG ======================
TRAIN_PATH    = "data/embeddings/train2025_sentiment_with_emb.csv"   # ← từ bước 4
TEST_PATH     = "data/embeddings/test2025_sentiment_with_emb2.csv"   # ← từ bước 4
EMBEDDING_DIM = 1024

# Dynamic Multi-Seed Ensemble
TRAIN_SEEDS    = [30, 31, 32]     # ← tăng để ensemble nhiều hơn
BACKTEST_SEEDS = [100]
NUM_CPU        = 4

# CVaR reward config
CVAR_CONFIG = {'enable': True, 'weight': 0.5, 'window': 90, 'alpha': 0.05}

# Models được lưu vào:
# → models/trained_seeds/   (mỗi seed 1 file .zip)

# ====================== BACKTEST CONFIG ======================
TRAINED_MODEL_DIRS = [
    "models/trained_seeds",   # ← thư mục chứa model đã train
]

# Output backtest:
# → results/allocation_{METHOD_NAME}.csv
# → results/history_{METHOD_NAME}.csv
```

---

### Bước 6: Kiểm định mô hình DRL

Mở `notebooks/06_kiem_dinh_drl.ipynb`

```python
# ====================== KIỂM ĐỊNH XU HƯỚNG (CAPM + LASSO) ======================
VNM_PATH   = "results/vnm.csv"                                     # ← VNM ETF proxy
STOCK_PATH = "data/processed/testvn_risk.xlsx"                     # ← dữ liệu test có chỉ báo
ALLOC_PATH = "results/allocation_5_L110m_N10_L21m_K1_opent+1.csv"  # ← từ bước 5

# ====================== KIỂM ĐỊNH TƯƠNG LAI (IG ALIGNMENT) ======================
IG_PATH = "results/ig_explanation_5_L110m_N10_L21m_K1 (2).csv"    # ← Integrated Gradients

BETA_PATHS = {
    "1-day":  "results/beta_single_step.csv",
    "5-day":  "results/beta_multi_step_W5.csv",
    "20-day": "results/beta_multi_step_W20.csv",
    "60-day": "results/beta_multi_step_W60.csv",
}

# ====================== PHÂN TÍCH RỦI RO CỔ PHIẾU ======================
df = pd.read_excel("data/processed/test2025_risk.xlsx")
# → Tính Sharpe, Sortino, CVaR, Max Drawdown, Calmar từng mã
```

Notebook này kiểm định:
- **Xu hướng**: CAPM rolling 30 ngày, LASSO feature selection, phân loại chế độ thị trường (Bull/Bear/Sideways)
- **Tương lai (Lookahead)**: Tương quan Pearson giữa Integrated Gradients và Beta hồi quy chéo, kiểm định t-test có ý nghĩa thống kê
- **Rủi ro**: Tính đầy đủ các chỉ số quản lý rủi ro cho từng cổ phiếu

---

## 🛠️ Cài Đặt

```bash
git clone https://github.com/<your-username>/DRL-Portfolio.git
cd DRL-Portfolio
pip install -r requirements.txt
```

**requirements.txt** (các thư viện chính):

```
torch>=2.0.0
transformers>=4.38
stable-baselines3
sb3-contrib
finrl
gymnasium
pandas
numpy
scikit-learn
openpyxl
vncorenlp
underthesea
selenium
newspaper3k
beautifulsoup4
tqdm
quantstats
statsmodels
scipy
```

> 💡 **Khuyến nghị**: Dùng Google Colab Pro (A100) cho bước 3–5, Kaggle GPU cho backtest.

---

## 🧠 Kiến Trúc Mô Hình

### Recurrent PPO-LSTM

```
Observation Space:
  [Giá OHLCV] + [8 Chỉ báo kỹ thuật] + [PhoBERT Embedding (1024-d)]
       ↓
  LSTM Layer (Memory across time steps)
       ↓
  Policy Head → Action (Tỷ trọng danh mục)
  Value Head  → V(s) (Ước lượng giá trị)
       ↓
  CVaR Reward Penalty:
    r_t = return_t − λ · CVaR(α=0.05, window=90)
```

### Dynamic Multi-Seed Ensemble

```
30 Seeds × Training
      ↓
Layer 1: Lọc pool dài hạn (10 tháng rolling Sharpe)
      ↓
Layer 2: Chọn model tốt nhất ngắn hạn (1 tháng)
      ↓
Final Allocation = Weighted Average
```

### PhoBERT Multi-task

```
Input: Tiêu đề + Lead paragraph (sau entity masking + VnCoreNLP)
      ↓
PhoBERT-large (vinai/phobert-large)
      ↓
[CLS] embedding (1024-d) → Sentiment Score [-1, +1]
                         → Task 2: Relevance / Impact
```

---

## 📈 Chi Tiết Kết Quả Kiểm Định

### Kiểm định Lookahead Bias (IG Alignment)

| Horizon | Pearson r | p-value | Kết luận |
|---------|-----------|---------|----------|
| 1-day   | > 0       | < 0.05  | ✅ Có tương quan |
| 5-day   | > 0       | < 0.05  | ✅ Có tương quan |
| 20-day  | > 0       | < 0.05  | ✅ Có tương quan |
| 60-day  | > 0       | < 0.05  | ✅ Có tương quan |

> Model học được tín hiệu thật từ dữ liệu, không bị lookahead bias.

### Phân tích chế độ thị trường

Composite score từ 3 tín hiệu (tránh MSM collapse trên high-drift market):
- `trend_20`: Rolling mean return 20 ngày
- `vol_20`: Rolling std 20 ngày
- `drawdown`: Drawdown từ đỉnh 6 tháng

---

## 📚 Tài Liệu Tham Khảo

- [FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading](https://github.com/AI4Finance-Foundation/FinRL)
- [PhoBERT: Pre-trained language models for Vietnamese](https://github.com/VinAIResearch/PhoBERT)
- [Recurrent PPO (sb3-contrib)](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)

---

## 👤 Tác Giả

**Lê Chí Bảo**  
Sinh viên năm cuối FinTech · Đại học Ngân hàng TP.HCM  
📧 0772288380chibaot@gmail.com  
🔗 [GitHub](https://github.com/lechibao-quant)

---

## 📄 License

MIT License — Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

> ⚠️ **Disclaimer**: Dự án này phục vụ mục đích nghiên cứu học thuật. Không phải lời khuyên đầu tư tài chính.
