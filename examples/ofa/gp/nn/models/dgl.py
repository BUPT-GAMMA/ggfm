import torch
from dgl.nn.pytorch.conv import GINConv, RelGraphConv

from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.models.util_model import MLP


class DGLGIN(MultiLayerMessagePassing):
    """Graph Isomorphism Network (GIN) implementation using DGL.
    
    This class implements a multi-layer GIN model with customizable architecture
    using the Deep Graph Library (DGL) backend.
    
    Args:
        num_layers (int): Number of GIN layers
        inp_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        drop_ratio (float, optional): Dropout ratio. Defaults to 0
        JK (str, optional): Jumping knowledge mode ("last", "concat", "max", "sum"). Defaults to "last"
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True
        
    Attributes:
        conv (nn.ModuleList): List of GIN convolution layers
    """

    def __init__(
        self,
        num_layers,
        inp_dim,
        out_dim,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.build_layers()

    def build_input_layer(self):
        """Constructs the input GIN layer.
        
        Returns:
            GINConv: Input layer with MLP as aggregator
        """
        return GINConv(
            MLP(
                [self.inp_dim, 2 * self.inp_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            learn_eps=True,
        )

    def build_hidden_layer(self):
        """Constructs a hidden GIN layer.
        
        Returns:
            GINConv: Hidden layer with MLP as aggregator
        """
        return GINConv(
            MLP(
                [self.out_dim, 2 * self.out_dim, self.out_dim],
                batch_norm=self.batch_norm is not None,
            ),
            learn_eps=True,
        )

    def build_message_from_input(self, g, input_feat="feat"):
        """Builds input message dictionary for message passing.
        
        Args:
            g (DGLGraph): Input graph
            input_feat (Union[str, torch.Tensor]): Input node features or feature key
            
        Returns:
            dict: Message dictionary containing graph and node features
            
        Raises:
            NotImplementedError: If input_feat type is not supported
        """
        if isinstance(input_feat, str):
            h = g.ndata[input_feat]
        elif torch.is_tensor(input_feat):
            h = input_feat
        else:
            raise NotImplementedError("Not supported input type")
        return {"g": g, "h": h}

    def build_message_from_output(self, g, h):
        """Builds output message dictionary for message passing.
        
        Args:
            g (DGLGraph): Input graph
            h (torch.Tensor): Node features
            
        Returns:
            dict: Message dictionary containing graph and node features
        """
        return {"g": g, "h": h}

    def layer_forward(self, layer, message):
        """Forward pass through a single layer.
        
        Args:
            layer (int): Layer index
            message (dict): Message dictionary containing graph and features
            
        Returns:
            torch.Tensor: Output features from the layer
        """
        return self.conv[layer](message["g"], message["h"])


class DGLRGCN(MultiLayerMessagePassing):
    """Relational Graph Convolutional Network implementation using DGL.
    
    This class implements a multi-layer RGCN model with support for multiple relation
    types using the Deep Graph Library (DGL) backend.
    
    Args:
        num_layers (int): Number of RGCN layers
        num_rels (int): Number of relation types
        inp_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        num_bases (int, optional): Number of bases for basis decomposition. Defaults to 4
        drop_ratio (float, optional): Dropout ratio. Defaults to 0
        JK (str, optional): Jumping knowledge mode ("last", "concat", "max", "sum"). Defaults to "last"
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True
        
    Attributes:
        num_rels (int): Number of relation types
        num_bases (int): Number of bases for weight decomposition
        conv (nn.ModuleList): List of RGCN convolution layers
    """

    def __init__(
        self,
        num_layers,
        num_rels,
        inp_dim,
        out_dim,
        num_bases=4,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.build_layers()

    def build_input_layer(self):
        """Constructs the input RGCN layer.
        
        Returns:
            RelGraphConv: Input RGCN layer
        """
        return RelGraphConv(
            self.inp_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_hidden_layer(self):
        """Constructs a hidden RGCN layer.
        
        Returns:
            RelGraphConv: Hidden RGCN layer
        """
        return RelGraphConv(
            self.out_dim, self.out_dim, self.num_rels, num_bases=self.num_bases
        )

    def build_message_from_input(self, g):
        """Builds input message dictionary for message passing.
        
        Args:
            g (DGLGraph): Input graph
            
        Returns:
            dict: Message dictionary containing graph, node features and edge types
        """
        return {"g": g, "h": g.ndata["feat"], "e": g.edata["type"]}

    def build_message_from_output(self, g, h):
        """Builds output message dictionary for message passing.
        
        Args:
            g (DGLGraph): Input graph
            h (torch.Tensor): Node features
            
        Returns:
            dict: Message dictionary containing graph, node features and edge types
        """
        return {"g": g, "h": h, "e": g.edata["type"]}

    def layer_forward(self, layer, message):
        """Forward pass through a single layer.
        
        Args:
            layer (int): Layer index
            message (dict): Message dictionary containing graph and features
            
        Returns:
            torch.Tensor: Output features from the layer
        """
        return self.conv[layer](message["g"], message["h"], message["e"])
