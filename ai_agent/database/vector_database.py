##########
# Import #
##############################################################################

import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType

from .__base import BaseDatabase
from ai_agent.embedding.__base import BaseEmbedding
from ai_agent.utilities.logger import get_utc_logger

##############################################################################

class QdrantVectorDatabase(BaseDatabase):
    def __init__(self, logger: logging.Logger = None, timeout: int=120) -> None:
        self.qdrant_client = QdrantClient(
            url = os.getenv('QDRANT_URL'),
            api_key = os.getenv('QDRANT_API_KEY'),
            timeout=timeout
        )
        
        if logger is None:
            logger = get_utc_logger(__name__, logging.INFO)
        self.logger = logger
    
    ##########################################################################
    
    def ingest_data(
            self,
            data_points: list[dict[str, any]],
            collection_name: str,
            wait_ingest: bool = False
    ) -> None:
        """
        Ingests text and their embeddings into a Qdrant collection.

        Parameters
        ----------
        data_points: List[Dict[str, Any]]:
            A list of dictionaries, where each dictionary contains
            "raw_text", "embedding", and "meta_data" with "article_id" 
            and "created_at".
        collection_name: str:
            The name of the Qdrant collection to use.
        """
        if not data_points:
            self.logger.info("No data points to ingest.")
            return

        # Prepare points for ingestion, now with extra metadata.
        points = []
        for i, data_point in enumerate(data_points):
            unique_chunk_id = str(uuid.uuid4())
            
            # Extract the necessary data from the dictionary
            raw_text = data_point.get("raw_text")
            embedding = data_point.get("embedding")
            meta_data = data_point.get("meta_data", {})

            # Ensure we have all the required fields
            if raw_text is None or embedding is None:
                self.logger.info(
                    f"Skipping data point at index {i} due to missing 'raw_text' or 'embedding'."
                )
                continue

            # Add a point to the list
            points.append(
                PointStruct(
                    # Using a unique ID for each chunk. A good approach might be a hash
                    # of the raw_text or a combination of article_id and chunk index.
                    # For simplicity, using the index `i` for this example.
                    id=unique_chunk_id, 
                    vector=embedding,
                    payload={
                        "raw_text": raw_text,
                        "article_id": meta_data.get("article_id"),
                        "created_at": meta_data.get("created_at")
                    }
                )
            )

        # Check if there are any points to upsert
        if not points:
            self.logger.info("No valid points to ingest.")
            return

        self.logger.info(
            f"Adding {len(points)} points to Qdrant collection '{collection_name}' with metadata..."
        )
        
        self.qdrant_client.upsert(
            collection_name=collection_name,
            wait=wait_ingest,
            points=points,
        )
        self.logger.info("Ingestion successful.")
        
    ##########################################################################
    
    def query_data(
            self,
            query_text: str,
            embedding_client: BaseEmbedding,
            collection_name: str,
            top_k: int = 5,
            least_score: float = 0.5
    ) -> list[dict[str, any]]:
        """
        Queries the Qdrant collection for similar documents.

        Parameters
        ----------
        query_text: str:
            The text to search for.
        collection_name: str:
            The name of the Qdrant collection to query.
        top_k: int:
            The number of top results to return. Defaults to 5.

        Returns
        -------
        List[Dict[str, Any]]:
            A list of the search results, including the raw text and metadata.
        """
        if not query_text:
            self.logger.info("Query text cannot be empty.")
            return []

        # Step 1: Get the embedding for the query text.
        # The embedder function expects a list of strings, so we wrap the query.
        query_embeddings = embedding_client.get_text_embeddings(texts=[query_text])

        if not query_embeddings:
            self.logger.info("Failed to generate embedding for the query text.")
            return []

        # The query_embeddings will be a list of lists. We need the first one.
        query_vector = query_embeddings[0]['embedding']

        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,  # This ensures we get the original text and metadata back
        )

        # Step 3: Format the results for a clean output.
        results = []
        for result in search_results:
            if result.score >= least_score:
                results.append(
                    {
                        "raw_text": result.payload.get("raw_text"),
                        "article_id": result.payload.get("article_id"),
                        "score": result.score,
                        "created_at": result.payload.get("created_at")
                    }
                )
        
        return results
    
    ##########################################################################
    
    def get_by_article_id(
        self,
        article_id: str,
        collection_name: str
    ) -> list[dict[str, any]]:
        """
        Retrieves all points from a Qdrant collection that match a specific article_id.

        This method uses Qdrant's scroll functionality to efficiently retrieve all
        matching points without a vector similarity search.

        Parameters
        ----------
        article_id: str:
            The unique identifier of the article to retrieve.
        collection_name: str:
            The name of the Qdrant collection.

        Returns
        -------
        List[Dict[str, Any]]:
            A list of all the points found with the matching article_id,
            including their raw text and metadata.
        """
        if not article_id:
            self.logger.info("Article ID cannot be empty.")
            return []
        
        # Create a filter to match the nested article_id
        article_filter = Filter(
            must=[
                FieldCondition(
                    key="article_id",
                    match=MatchValue(value=article_id),
                )
            ]
        )
        
        self.logger.info(article_filter)

        # Use the scroll method to get all points that match the filter
        scroll_results, _ = self.qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=article_filter,
            limit=100,  # Set a reasonable limit, or use pagination for large results
            with_payload=True,
            with_vectors=False
        )
        
        # Format the results for a clean output
        results = []
        for result in scroll_results:
            results.append(
                {
                    "raw_text": result.payload.get("raw_text"),
                    "article_id": result.payload.get("meta_data", {}).get("article_id"),
                    "created_at": result.payload.get("meta_data", {}).get("created_at")
                }
            )
        print(f"results: {results}")
        return results

    ##########################################################################
    
    def create_article_id_index(self, collection_name: str, field: str) -> None:
        """
        Creates a payload index for the field.
        This is required for efficient filtering on this field.
        """
        try:
            self.qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field,  # Use dot notation for nested fields
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self.logger.info(f"Index created for '{field}' in collection '{collection_name}'.")
        except Exception as e:
            self.logger.info(f"Error creating index: {e}")
    
##############################################################################
