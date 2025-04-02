# Fine-Tuning-For-Sense-Enhancement

This repository contains the code for the submission "On the Relation Between Fine-Tuning, Topological Properties, and Task Performance in Sense-Enhanced Embeddings".

The code has following functionalities:
1) Fine-tuning BERT using different methods (SCL, SPL, and Task adaptation)
2) Evaluation of a model's embedding space in terms of _topology_, and _WiC task performance_
3) Evaluation of embedding spaces after isotropization in terms of _topology_, and _WiC task performance_

## Usage Example

### Fine-Tuning

#### Argument Definitions

- **finetuning_method (str):** Method of fine-tuning (SCL, SPL or TASK)
- **learning_rate (float):** Learning Rate
- **epoch (int):** Number of Epochs
- **tau (float) (optional):** Temperature of the loss. Required with SCL and SPL.

#### Example:

```bash
python train_main.py TASK 0.0001 20 
```

```bash
python train_main.py SCL 0.0001 20 -tau 0.1
```

### Evaluation

#### Argument Definitions

- **model_type**: Type of model (BERT or fine-tuned for SCL, SPL or TASK)
- **tau**: tau for SCL or SPL
- **isotropized**: Temperature of the loss (only for SCL and SPL)

#### Example:

For evaluating BERT's embedding space:
```bash
python eval_main.py BERT  
```

For evaluating a fine-tuned model's embedding space:
```bash
python eval_main.py SCL 0.1 
```


For evaluating istoropized BERT embeddings:
```bash
python eval_main.py BERT --isotropized
```
