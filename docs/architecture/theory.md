# Theory & Methods

This section describes the mathematical formulations and algorithms underlying the Neural-LAM graph models.

## The Encode-Process-Decode Paradigm

All graph-based models in Neural-LAM (`GraphLAM`, `HiLAM`, and `HiLAMParallel`) follow the classic "Encode-Process-Decode" paradigm for graph neural networks.

1. **Encode**: The input grid state \( X^t \) at timestep \( t \) is mapped to latent node features on the graph:
   \[ H_{grid} = \text{Encoder}(X^t) \]

2. **Process**: A series of message passing steps updates the latent node representations based on the graph connectivity:
   \[ H'_{grid} = \text{Processor}(H_{grid}, \mathcal{G}) \]

3. **Decode**: The updated latent features are mapped back to predict the residual change for the next timestep:
   \[ \Delta X^{t+1} = \text{Decoder}(H'_{grid}) \]
   \[ X^{t+1} = X^t + \Delta X^{t+1} \]

## Message Passing Functions

Neural-LAM supports several GNN layers for message passing along different edge types (e.g., Grid-to-Mesh, Mesh-to-Mesh).

### InteractionNet

The default layer is the `InteractionNet` (based on Interaction Networks). Given a sender node \( v_s \), a receiver node \( v_r \), and an edge feature \( e_{s,r} \), the message passing works as follows:

1. **Edge Update (Message Formulation)**:
   \[ m_{s,r} = \text{MLP}_{edge}\left([h_s, h_r, e_{s,r}]\right) \]

2. **Node Update (Aggregation)**:
   \[ h'_r = \text{MLP}_{node}\left([h_r, \sum_{s \in \mathcal{N}(r)} m_{s,r}]\right) \]

### PropagationNet

`PropagationNet` modifies the Interaction Network to strongly incentivize directional information flow from senders to receivers, which is crucial for moving information up and down the hierarchical mesh levels:

\[ h'_r = h_r + \text{MLP}_{prop}\left([h_r, \sum_{s \in \mathcal{N}(r)} m_{s,r}]\right) \]

This residual connection ensures that nodes maintain their internal state while selectively incorporating new information from neighbors.

## Loss Weighting

To balance predictions across variables with different physical scales and variances, the loss function uses a weighted Mean Squared Error (MSE):

\[ \mathcal{L} = \frac{1}{V \cdot N} \sum_{v=1}^V \sum_{i=1}^N w_v \cdot \left( \hat{X}^{t+1}_{i,v} - X^{t+1}_{i,v} \right)^2 \]

Where \( w_v \) are variable-specific weights defined in the `loss_weighting` module, and the errors are computed on standardized variables.
