# diffusion2vec
Reference implementation of Diffusion2Vec.

Diffusion to vector is a transductive topological graph embedding procedure inspired by sequence based graph embedding procedures. It is robust to graph densification and growth. The created embeddings are competitive with other sequence based embedding methods (e.g. DeepWalk/Node2Vec). It can be used to learn embeddings of networks with millions of nodes and it allows for distributed processing in the graph pre-processing, sequence generation and learning phases.

The hyperparameters of the model are:



