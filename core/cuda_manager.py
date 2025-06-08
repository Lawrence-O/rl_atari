import torch

class CudaStreamManager:
    """Manages CUDA streams for parallel processing of multiple games"""
    
    def __init__(self, game_names):
        self.streams = {}
        self.events = {}
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}

        for game in game_names:
            self.streams[game] = torch.cuda.Stream()
            self.events[game] = torch.cuda.Event(enable_timing=True)
    
    def process_batch_with_streams(self, model, game_batches):
        """Process batches from different games in parallel using CUDA streams"""
        results = {}
        
        # Launch each game's processing in its own stream
        for game, batch in game_batches.items():
            # Work in game-specific stream
            with torch.cuda.stream(self.streams[game]):
                # Process this game's batch
                results[game] = model(batch, game)
                # Record completion event
                self.events[game].record(self.streams[game])
        
        # Wait for all streams to complete
        for event in self.events.values():
            event.synchronize()
            
        return results
    
    def capture_backbone_graph(self, backbone, input_shape, batch_size=32):
        """Capture CUDA graph for backbone forward pass"""
        # Create static input/output for the graph
        static_input = torch.zeros(batch_size, *input_shape, device='cuda')
        
        # Warmup before capture
        backbone(static_input)
        torch.cuda.synchronize()
        
        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = backbone(static_input)
        
        self.graphs['backbone'] = g
        self.static_inputs['backbone'] = static_input
        self.static_outputs['backbone'] = static_output
        
        return g
    
    def run_backbone_with_graph(self, input_batch):
        """Run backbone forward pass using captured graph"""
        # Copy input data to static tensor
        self.static_inputs['backbone'].copy_(input_batch)
        # Replay the graph
        self.graphs['backbone'].replay()
        # Return a clone of the output
        return self.static_outputs['backbone'].clone()
            
    
    def get_next_stream(self):
        """Get the next CUDA stream"""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream