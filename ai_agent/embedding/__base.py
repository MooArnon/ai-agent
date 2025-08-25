##########
# Import #
##############################################################################

from abc import abstractmethod

##############################################################################

class BaseEmbedding:
    def __init__(self):
        pass
    
    ##########################################################################
    
    @abstractmethod
    def get_text_embeddings() -> None:
        raise NotImplementedError(
            "The child class must implement this method"
        )
        
    ##########################################################################
    
    @abstractmethod
    def find_closest_text() -> None:
        raise NotImplementedError(
            "The child class must implement this method"
        )
    
    ##########################################################################
    
##############################################################################
