#this file converts a scikit neural network into an OMLT model.
#NOTE: this requires OMLT v0.3.0
#TODO: update for OMLT 1.0

from omlt.neuralnet import NetworkDefinition

#Convert a sci-kit MLP regressor into an OptML network definition
def load_scikit_mlp(nn,scaling_object=None,input_bounds=None):
    n_inputs = nn.n_features_in_
    n_outputs = nn.n_outputs_
    node_id_offset = n_inputs
    layer_offset = 0
    w = dict()
    b = dict()
    a = dict()
    n_layers = nn.n_layers_

    for l in range(n_layers-1):
        weights = nn.coefs_[l]
        biases = nn.intercepts_[l]
        n_layer_inputs, n_layer_nodes = weights.shape
        for i in range(n_layer_nodes):
            layer_w = dict()
            for j in range(n_layer_inputs):
                layer_w[j+layer_offset] = weights[j,i]
            w[node_id_offset] = layer_w
            b[node_id_offset] = biases[i]
            if l == n_layers - 2: #this is the output layer
            	if nn.out_activation_ == 'identity':
            		a[node_id_offset] = 'linear' 
            	else:
            		a[node_id_offset] = nn.out_activation_
            else:
            	a[node_id_offset] = nn.activation
            node_id_offset += 1
        layer_offset += n_layer_inputs
    n_nodes = len(a) + n_inputs
    n_hidden = n_nodes - n_inputs - n_outputs
    
    return NetworkDefinition(n_inputs=n_inputs,
                              n_hidden=n_hidden,
                              n_outputs=n_outputs,
                              weights=w,
                              biases=b,
                              activations=a,
                              scaling_object=scaling_object,
                              input_bounds=input_bounds
                            )