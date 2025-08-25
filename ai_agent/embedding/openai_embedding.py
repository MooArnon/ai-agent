##########
# Import #
##############################################################################

from datetime import datetime, timezone
import logging
from typing import List, Tuple, Optional
import uuid

import numpy as np
import openai
from scipy.spatial.distance import cosine

from .__base import BaseEmbedding
from ai_agent.utilities.logger import get_utc_logger

##############################################################################

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, logger=None):
        self.client = openai.OpenAI()
        if logger is None:
            logger = get_utc_logger(__name__, logging.INFO)
        self.logger = logger
    
    ##########################################################################

    def get_text_embeddings(
            self,
            texts: List[str],
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            model: str = "text-embedding-3-small",
    ) -> List[dict[str, any]]:
        """
        Generates text embeddings for a list of strings, with chunking.

        Parameters
        ----------
        texts: List[str]:
            A list of strings to embed. Each string will be chunked.
        article_id: str:
            A unique identifier for the article. This will be the same for all
            chunks derived from the same original text.
        chunk_size: int:
            The maximum number of characters per chunk.
        chunk_overlap: int:
            The number of characters to overlap between chunks.
        model: str:
            The name of the embedding model to use. 
            Defaults to "text-embedding-3-small".

        Returns
        -------
        List[Dict[str, Any]]:
            A list of dictionaries, where each dictionary represents a chunk
            with its metadata, raw text, and embedding vector.
        """
        article_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            self.logger.info("Error: Input must be a list of strings.")
            return []

        # A simple character-based chunking logic.
        # More sophisticated methods might use tokenizers 
        # or sentence splitters.
        all_chunks = []
        for text in texts:
            
            # Simple chunking by character count
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                all_chunks.append(chunk)
                start += chunk_size - chunk_overlap

        if not all_chunks:
            return []

        response = self.client.embeddings.create(
            input=all_chunks,
            model=model
        )

        # Combine the chunks, metadata, and embeddings 
        # into the desired format.
        results = []
        for i, chunk in enumerate(all_chunks):
            
            # Ensure the embedding exists for the current chunk.
            if i < len(response.data):
                embedding = response.data[i].embedding
                results.append({
                    "meta_data": {
                        "article_id": article_id,
                        "created_at": created_at,
                    },
                    "raw_text": chunk,
                    "embedding": embedding,
                })
        return results

    ##########################################################################
    
    def find_closest_text(
            self,
            query_embedding: List[float], 
            text_embedding_pairs: List[
                Tuple[
                    str, List[float]
                ]
            ]
    ) -> Optional[Tuple[str, float]]:
        """
        Finds the text with the most similar embedding to a query embedding 
        from a provided list.

        This function simulates converting a vector back to text 
        by finding the best textual match from a known set of options.

        Parameters
        ----------
        query_embedding (List[float]): 
            The embedding vector for the text you want to find matches for.
        text_embedding_pairs (List[Tuple[str, List[float]]]): 
            A list of tuples, where each tuple contains a string 
            and its corresponding embedding vector.

        Returns
        -------
            Optional[Tuple[str, float]]: 
            A tuple containing the most similar text and its similarity score
            (cosine similarity, where 1 is most similar), 
            or None if no match is found.
        """
        if not text_embedding_pairs:
            self.logger.info("Error: No text/embedding pairs provided to search.")
            return None

        # Convert the query embedding to a NumPy array for efficient calculation.
        query_np = np.array(query_embedding)

        # Initialize variables to track the best match.
        best_match_text = None
        
        # Start with a very low score for cosine similarity
        best_match_score = -1.0  

        for text, embedding in text_embedding_pairs:
            
            # Convert the current embedding to a NumPy array.
            embedding_np = np.array(embedding)

            # Calculate the cosine similarity between the two vectors.
            # The formula is 1 - cosine_distance.
            # A higher score means more similar.
            similarity = 1 - cosine(query_np, embedding_np)

            # If this is the best match so far, update the variables.
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_text = text

        # Return the best match and its score.
        return (best_match_text, best_match_score)
    
    ##########################################################################
    
##############################################################################
