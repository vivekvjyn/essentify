# Essentify

## How to run

**Create and activate conda environment**

```bash
conda create -n ess python=3.10
conda activate ess
```

**Install CUDA and CuDNN**

```bash
conda install -c conda-forge -y cudatoolkit=11.2 cudnn=8.1
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run analysis**
- Add you music collection to `/audio`.
- Run `main.py`

```bash
CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 python main.py
```

**Explore your music using Essentify**

```bash
streamlit run essentify.py
```

## Analysis report
[**View analysis report on MusAV**](https://github.com/enter-opy/essentify/blob/main/notebooks/analysis.ipynb)
