This is the code base for HW 4 of LING 384/784, Computational Psycholinguistic, Spring 2025, taught by Tom McCoy.

# Structure of the codebase
In the main directory:
- `configs/`: contains the configuration `.yaml` files that defines each run of the experiment;
- `my_models/`: stores folders each containing the `.pt` file for the trained model and the corresponding experiment configurations;
- `pcfg/`: stores PCFG `.txt` files to use;

Remainings are modules:
- `pcfg_dataset.py`: contains classes parsing a PCFG `.txt` file and convert to datasets for training and evaluation;
- `model.py`: the transformer architecture, containing pure sampling generation with log probabilities;
- `training.py`: training and evaluation function;
- `run_and_eval.py`: the **main** function to run;

# How to run
1. Define the relevant parameters in the `configs/default_config.yaml` file or some customized configs;
2. In terminal, after activating the environment and `cd` to the home directory, run: `python run_and_eval.py --config configs/default_config.yaml`;
3. The pure sampling generation sentences and their logProbs will be printed;
4. The trained model will be saved in `my_models/`; and the **main** function contains code to load the model and run generations;