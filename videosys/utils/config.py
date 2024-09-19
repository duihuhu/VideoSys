from typing import List
class DeployConfig:
    """Deploy configuration
    Args:
        enable_separate: use P/D separate mode or not
        role: the role of the engine is dit or vae when use separate model iteration
     
    """
    def __init__(
        self,
        enable_separate: bool=False,
        role: str=None,
        deploy_host: str = None,
        deploy_port: str = None,
        ) -> None: 
            self.enable_separate = enable_separate
            self.role = role
            self.global_ranks = None
            
            self.deploy_host = deploy_host
            self.deploy_port = deploy_port
            self._verify_args()
    
    def _verify_args(self) -> None:
        if self.enable_separate and self.role not in ['DIT', 'VAE']:
            raise ValueError(f"role of DiT Engine Instance must be prompt or decoder in separate mode")

    def set_global_ranks(self, global_ranks: List[int]) -> None:
        self.global_ranks = global_ranks
        self.global_ranks.sort()
        
    def get_global_ranks(self) -> List[int]:
        return self.global_ranks
