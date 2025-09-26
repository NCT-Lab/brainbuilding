# State Machine Configurations

This directory contains state machine configuration files (`.yaml`) for different scenarios and datasets.

## `_old` Configurations

These configurations are for use with the older training dataset, which has a sampling frequency of 500Hz and a specific experiment script format.

-   `state_config_old.yaml`: Used for running inference tests on the old dataset.
-   `state_config_old_train.yaml`: Used for training models on the old dataset.

## `_new` Configurations

These configurations are for the new dataset and experiment script, which uses a 250Hz sampling frequency.

-   `state_config_new.yaml`: Defines the standard inference scenario. In this configuration, predictions are only made during specific phases when the user is expected to be imagining motor movements or resting.
-   `state_config_new_inference_all.yaml`: A variation for inference that runs predictions in all possible phases of the experiment script.
