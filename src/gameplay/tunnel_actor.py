from src.core.actor import Actor
from src.rl.tunnel import TunnelConfig
import numpy as np

class TunnelActor(Actor):
    """
    An actor that represents the cylindrical tunnel environment.
    """
    def __init__(self, name="Tunnel", config=None, **kwargs):
        # Position is technically the start of the tunnel or center, 
        # but the config dictates the geometry.
        # We can use (0,0,0) as base and let drawing logic handle offsets based on config.
        super().__init__(name=name, x=0.0, y=0.0, z=0.0)
        
        if config:
            self.config = config
        else:
            # 使用 tunnel.py 中 TunnelConfig 的默认值
            self.config = TunnelConfig(**kwargs)

    def update(self, dt):
        # Tunnel is static
        pass
