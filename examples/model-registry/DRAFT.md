# Two models, traced back to their training data

The iris classifier registered as `iris-clf` (alias `champion`) scores
**1.0** accuracy on its held-out split.
<!-- claim: model-registry/82891d65b22e4e838d8a0cf49f773f2f metric=accuracy value=1.0 model=models:/iris-clf@champion -->

The tiny GPT-2 saved in `hf/model` finished training at a mean loss of
**2.1381**.
<!-- claim: model-registry/toy-seq-gpt2-0 metric=train_loss value=2.1381 tol=0.00005 model=hf/model -->

A claim can be right about its number and wrong about its model: this one
credits the GPT-2 with the iris run's accuracy of **1.0**.
<!-- claim: model-registry/82891d65b22e4e838d8a0cf49f773f2f metric=accuracy value=1.0 model=hf/model -->
