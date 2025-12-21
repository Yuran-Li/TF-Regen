"""
OpenAI 嵌入模型包装器
提供与 SentenceTransformer 兼容的接口
"""

from openai import OpenAI
import numpy as np
from typing import List, Union
import os
from tqdm import tqdm

class OpenAIEmbedder:
    """OpenAI 嵌入模型包装器，提供与 SentenceTransformer 兼容的接口"""
    
    def __init__(self, model_name: str = "text-embedding-3-large", device: str = "cpu"):
        """
        初始化 OpenAI 嵌入模型
        
        Args:
            model_name: 模型名称 (text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002)
            device: 设备参数（为了兼容性保留，实际不使用）
        """
        self.model_name = model_name
        
        # 从环境变量获取 API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        # 创建 OpenAI 客户端（新版 API）
        base_url = os.environ.get("OPENAI_API_BASE")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 100,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False
    ) -> np.ndarray:
        """
        编码文本为嵌入向量
        
        Args:
            texts: 文本列表或单个文本
            batch_size: 批次大小（OpenAI API 建议不超过 2048）
            show_progress_bar: 是否显示进度条
            normalize_embeddings: 是否归一化嵌入向量
            
        Returns:
            嵌入向量数组
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        embeddings = []
        
        # 批次处理
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 调用 OpenAI API（新版本）
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                
                # 提取嵌入向量
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {e}")
                raise
        
        # 转换为 numpy 数组
        embeddings = np.array(embeddings)
        
        # 归一化（如果需要）
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # 如果输入是单个字符串，返回单个向量
        if single_input:
            return embeddings[0]
        
        return embeddings


# 使用示例
if __name__ == "__main__":
    # 设置 API key
    # export OPENAI_API_KEY="your-api-key"
    
    # 创建模型
    try:
        model = OpenAIEmbedder("text-embedding-3-large")
        
        # 编码文本
        texts = [
            "This is a test sentence.",
            "Another example text."
        ]
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(embeddings)
        print(f"Similarity matrix:\n{similarity}")
    except ValueError as e:
        print(f"Error: {e}")
        print("请先设置环境变量: export OPENAI_API_KEY='your-api-key'")
