"""
OpenAI embedding model wrapper
provide an interface compatible with SentenceTransformer
"""

from openai import OpenAI
import numpy as np
from typing import List, Union
import os
from tqdm import tqdm

class OpenAIEmbedder:
    """OpenAI embedding model wrapper, provide an interface compatible with SentenceTransformer"""
    
    def __init__(self, model_name: str = "text-embedding-3-large", device: str = "cpu"):
        """
        initialize OpenAI embedding model
        
        Args:
            model_name: model name (text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002)
            device: device parameter (for compatibility, actually not used)
        """
        self.model_name = model_name
        
        # get API key from environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        # create OpenAI client (new API)
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
        encode text to embedding vector
        
        Args:
            texts: text list or single text
            batch_size: batch size (OpenAI API suggests不超过 2048）
            show_progress_bar: whether to show progress bar
            normalize_embeddings: whether to normalize the embedding vector
            
        Returns:
            embedding vector array
        """
        # ensure input is a list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        embeddings = []
        
        # batch processing
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            try:
                # call OpenAI API
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                
                # extract embedding vector
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size + 1}: {e}")
                raise
        
        # convert to numpy array
        embeddings = np.array(embeddings)
        
        # normalize (if needed)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # if input is a single string, return a single vector
        if single_input:
            return embeddings[0]
        
        return embeddings


# example usage
if __name__ == "__main__":
    # set API key
    # export OPENAI_API_KEY="your-api-key"
    
    # create model
    try:
        model = OpenAIEmbedder("text-embedding-3-large")
        
        # encode text
        texts = [
            "This is a test sentence.",
            "Another example text."
        ]
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(embeddings)
        print(f"Similarity matrix:\n{similarity}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the environment variable: export OPENAI_API_KEY='your-api-key'")
