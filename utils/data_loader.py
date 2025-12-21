from datasets import load_dataset, get_dataset_config_names
from datasets import concatenate_datasets
import random
import os
import numpy as np

def set_data_loader_seed(seed=42):
    """设置数据加载器的随机种子"""
    random.seed(seed)
    np.random.seed(seed)

def load_data(dataset_name, split, seed=42):
    # 设置随机种子
    set_data_loader_seed(seed)
    
    # Set offline mode to avoid network requests when cache exists
    offline_mode = os.environ.get("HF_DATASETS_OFFLINE", "false").lower() == "true"
    ## math dataset
    if dataset_name == "gsm8k":
        try:
            if split == "train":
                # simple version
                dataset = load_dataset("gsm8k", "main", split="train", download_mode="reuse_dataset_if_exists")
                # socratic version
                # dataset = load_dataset("gsm8k", "socratic")
            elif split == "test":
                dataset = load_dataset("gsm8k", "main", split="test", download_mode="reuse_dataset_if_exists")
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                if split == "train":
                    dataset = load_dataset("gsm8k", "main", split="train")
                elif split == "test":
                    dataset = load_dataset("gsm8k", "main", split="test")
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=seed)
    if dataset_name == "gsm_hard":
        try:
            dataset_all = load_dataset("reasoning-machines/gsm-hard", split="train", download_mode="reuse_dataset_if_exists") ##only train set 1320
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                dataset_all = load_dataset("reasoning-machines/gsm-hard", split="train")
        
        # shuffle the full dataset first with fixed seed
        dataset_all = dataset_all.shuffle(seed=seed)
        
        ##split train and test
        if split == "train":
            dataset = dataset_all.select(range(0,800))
        elif split == "test":
            dataset = dataset_all.select(range(800, 1319))
    if dataset_name == "math":
        try:
            train_dataset = load_dataset("nlile/hendrycks-MATH-benchmark", "default", split="train", download_mode="reuse_dataset_if_exists")
            test_dataset = load_dataset("nlile/hendrycks-MATH-benchmark", "default", split="test", download_mode="reuse_dataset_if_exists")
            dataset = concatenate_datasets([train_dataset, test_dataset])
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                train_dataset = load_dataset("nlile/hendrycks-MATH-benchmark", "default", split="train")
                test_dataset = load_dataset("nlile/hendrycks-MATH-benchmark", "default", split="test")
                dataset = concatenate_datasets([train_dataset, test_dataset])
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=seed)
        print(f"load total length of dataset: {len(dataset)}")
        # choose the level of difficulty first
        dataset = dataset.filter(lambda x: x["level"] == 3)
        print(f"load total length of dataset after filtering: {len(dataset)}")
        # re-split the dataset after filtering
        if split in ["train", "test"]:
            split_dataset = dataset.train_test_split(test_size=0.4, seed=42)
            dataset = split_dataset[split]
    ## commonsense dataset
    if dataset_name == "strategyQA":
        try:
            dataset_all = load_dataset("tasksource/strategy-qa", split="train", download_mode="reuse_dataset_if_exists") ##only train set 2290
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                dataset_all = load_dataset("tasksource/strategy-qa", split="train")
        
        # shuffle the full dataset first with fixed seed
        dataset_all = dataset_all.shuffle(seed=seed)
        
        ##split train and test
        if split == "train":
            dataset = dataset_all.select(range(0,1890))
        elif split == "test":
            dataset = dataset_all.select(range(1890, 2290))
    if dataset_name == "bamboogle":
        try:
            dataset_all = load_dataset("chiayewken/bamboogle", split="test", download_mode="reuse_dataset_if_exists") ##only train set 125
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                dataset_all = load_dataset("chiayewken/bamboogle", split="test")
        
        # shuffle the full dataset first with fixed seed
        dataset_all = dataset_all.shuffle(seed=seed)
        
        ##split train and test
        if split == "train":
            dataset = dataset_all.select(range(0,25))
        elif split == "test":
            dataset = dataset_all.select(range(25, 125))
    ##symbolic dataset
    if dataset_name == "coin":
        try:
            if split == "train":
                dataset = load_dataset("skrishna/coin_flip", split="train", download_mode="force_redownload")
            elif split == "test":
                dataset = load_dataset("skrishna/coin_flip", split="test", download_mode="force_redownload")
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                if split == "train":
                    dataset = load_dataset("skrishna/coin_flip", split="train", download_mode="force_redownload")
                elif split == "test":
                    dataset = load_dataset("skrishna/coin_flip", split="test", download_mode="force_redownload")
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=seed)
    if dataset_name == "letter":
        try:
            if split == "train":
                dataset = load_dataset("ChilleD/LastLetterConcat", split="train", download_mode="reuse_dataset_if_exists")
            elif split == "test":
                dataset = load_dataset("ChilleD/LastLetterConcat", split="test", download_mode="reuse_dataset_if_exists")
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                if split == "train":
                    dataset = load_dataset("ChilleD/LastLetterConcat", split="train")
                elif split == "test":
                    dataset = load_dataset("ChilleD/LastLetterConcat", split="test")
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)
    ##specific-domain dataset
    if dataset_name == "legal":
        sub_task = "abercrombie"
        try:
            if split == "train":
                dataset = load_dataset(
                    "nguha/legalbench",
                    sub_task,
                    split="train",
                    download_mode="reuse_dataset_if_exists",
                    trust_remote_code=True,
                )
            elif split == "test":
                dataset = load_dataset(
                    "nguha/legalbench",
                    sub_task,
                    split="test",
                    download_mode="reuse_dataset_if_exists",
                    trust_remote_code=True,
                )
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                if split == "train":
                    dataset = load_dataset(
                        "nguha/legalbench",
                        sub_task,
                        split="train",
                        trust_remote_code=True,
                    )
                elif split == "test":
                    dataset = load_dataset(
                        "nguha/legalbench",
                        sub_task,
                        split="test",
                        trust_remote_code=True,
                    )
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)
    if dataset_name == "Headline":
        try:
            if split == "train":
                dataset = load_dataset("SaguaroCapital/sentiment-analysis-in-commodity-market-gold", split="train", download_mode="reuse_dataset_if_exists")
            elif split == "test":
                dataset = load_dataset("SaguaroCapital/sentiment-analysis-in-commodity-market-gold", split="test", download_mode="reuse_dataset_if_exists")
        except Exception as e:
            if offline_mode:
                print(f"Error loading dataset in offline mode: {e}")
                raise
            else:
                # If cache doesn't exist, try downloading
                if split == "train":
                    dataset = load_dataset("SaguaroCapital/sentiment-analysis-in-commodity-market-gold", split="train")
                elif split == "test":
                    dataset = load_dataset("SaguaroCapital/sentiment-analysis-in-commodity-market-gold", split="test")
        
        # shuffle dataset with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)
        # Check if dataset was loaded successfully
    if 'dataset' not in locals():
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets: gsm8k, gsm_hard, math, strategyQA, bamboogle, coin, letter, legal, Headline")

    return dataset