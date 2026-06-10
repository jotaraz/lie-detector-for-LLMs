"""
This script should be used to train and then compare a lot of different linear probes for deception.

# Very general
I will run the script we are about to write on a remote node that (A) has large GPUs so that everything can be run locally and (B) all the required activations are stored locally there. Here, on this machine there only is /Users/jo/code/Jinesis/deception-detection/autocomp2/data-autoconv4/meta-llama--Llama-3.1-8B-Instruct.pt, however you can also look at /Users/jo/code/Jinesis/deception-detection/autocomp2/basic-computations.ipynb to see how to interact with the activation tensors.
To see almost all potentially available activations see https://huggingface.co/datasets/jo-chen/lie-detector-results/tree/main/.
Furthermore another HuggingFace repository (https://huggingface.co/datasets/jo-chen/lying-detection-activations/tree/main/) contains trained probes, these probes where trained on the so-called 'repe' dataset, there you can find e.g., Qwen__Qwen2.5-32B-Instruct/probes/20260326_192417_logistic_layer16 -- a probe trained for Qwen2.5-32B-Instruct, which is to be used on layer 16. 

# Setup
This script should do/use the following:
Consider the (probed) models:
- Llama-3.1-8B-Instruct
- Llama-3.3-70B-Instruct
- Qwen2.5-32B-Instruct
- Qwen2.5-72B-Instruct
Then, for all probed models there are multiple datasets; these consist of a set of fixed system and user prompts, the probed model's response and a judge's label for this conversation (the conversation consists of system prompt, user prompt and model response). There usually are more than 2 labels the judge (typically gpt-5.4-mini) can give -- however, we specify two of them and use the activations the probed models have while generating responses that then get this label as the training data for the probes.
## Consider the following datasets (there are "Full" datasets on which we also train probes, and there are "Extra" datasets on which we only evaluate other probes):

### (Full) Dataset 1 "unprompted_roleplay" (autocomp2/data-more-roleplay)
- Scored files: for each probed model use the most recent scored files using gpt-5.4-mini as a judge, e.g., /Users/jo/code/Jinesis/deception-detection/autocomp2/data-more-roleplay/scored-20260608-162007-Qwen--Qwen2.5-32B-Instruct.json
- Labels: use the conversations with labels HONEST (as honest) and DECEPTIVE (as deceptive), ignore all other labels
### (Full) Dataset 2 "prompted_roleplay" (autocomp2/data-ir-honest and autocomp2/data-ir-dishonest)
- Scored files: for each probed model use the most recent scored files using gpt-5.4-mini as a judge, e.g., /Users/jo/code/Jinesis/deception-detection/autocomp2/data-ir-honest/meta-llama--Llama-3.3-70B-Instruct-judge-gpt-5.4-mini-20260609-034237.json
- Labels: use the conversations from data-ir-honest that were labeled "HONEST" (as honest) and the conversations from data-ir-dishonest that were labeled "DECEPTIVE" (as deceptive)
### (Full) Dataset 3 "company_assistant" (autocomp2/data-company-assistant-roleplay)
- Scored files: for each probed model use the most recent scored files using gpt-5.4-mini as a judge, e.g., /Users/jo/code/Jinesis/deception-detection/autocomp2/data-company-assistant-roleplay/meta-llama--Llama-3.1-8B-Instruct-judge-gpt-5.4-mini-96dc4d84-20260609-121843.json
- Labels: Use the conversations with labels "honest_authorized" (as honest) and "deceptive" (as "deceptive"), ignore all other labels

### Extra Dataset A "true stories" (autocomp2/data-true-stories)
-- These are unlabeled/unscored. We use this as a control dataset, i.e., we expect a good deception detection probe to classify all activations here as honest.
### Extra Dataset B "fake stories" (autocomp2/data-fake-stories)
-- These are unlabeled/unscored. We use this as a control dataset, i.e., we expect a good deception detection probe to classify all activations here as honest.
### Extra Dataset C "evasive compnay assistant" (autocomp2/data-company-assistant-roleplay)
- Scored files: for each probed model use the most recent scored files using gpt-5.4-mini as a judge, e.g., /Users/jo/code/Jinesis/deception-detection/autocomp2/data-company-assistant-roleplay/meta-llama--Llama-3.1-8B-Instruct-judge-gpt-5.4-mini-96dc4d84-20260609-121843.json
- Labels: Use the conversations with labels "evasive" for the control dataset.

The default should be, using the activations from each generated token in the different conversations as training data for the probes -- "honest" activations should be mapped to -1 by the probe and "deceptive" activations should be mapped to +1 by the probe.
Thus, for each of the 4 probed models, for each of the three (full) datasets there are two classes of conversations, and thus activations, honest and deceptive ones. For each model we have the activations for multiple layers (at most the first 60 layers, depending on the depth of the probed model).
Thus, we could define a dictionary like this:
activation_dict[model_name][layer_id][dataset_name][type] = (list of activations for this model, layer, dataset and type)
here, type is either "honest" or "deceptive".
We can then furthermore partition this into a train and a test split according to some split fraction (default 80/20) which should apply conversation wise -- e.g., the conversations of the first 80% of honest conversations are used in the train split.
This then defines the train_activation_dict and test_activation_dict such that 
train_activation_dict[model_name][layer_id][dataset_name][type] + test_activation_dict[model_name][layer_id][dataset_name][type] = activation_dict[model_name][layer_id][dataset_name][type], and
len(train_activation_dict[model_name][layer_id][dataset_name][type]) / len(test_activation_dict[model_name][layer_id][dataset_name][type]) \approx 4 (in the 80/20 example) -- note that the number of tokens in each conversation's response varies this can strongly deviate from a ratio of 4 in practice.

# Training and evaluation
Then, I want you to train the following probes:
For each model, each (available) layer, and each dataset please train a probe (on the train split).
The regularization coefficient of the trained linear probes should default to 10.0.
This probe should then be tested on this datasets test split, on the other (full) datasets test splits, and on the extra/control datasets.
Furthermore, please also train probes on the combination of datasets:
train_activation_dict[model_name][layer_id][dataset 1 + dataset 2][type] = train_activation_dict[model_name][layer_id][dataset 1][type][:n] + train_activation_dict[model_name][layer_id][dataset 2][type][:n],
here n is the length of the smaller training dataset (i.e., n = min(len(train_activation_dict[model_name][layer_id][dataset 1][type]), len(train_activation_dict[model_name][layer_id][dataset 2][type])) to ensure that both datasets are equally represented.
These 'combined' probes should then also be tested on each individual full dataset's test split, and on the extra/control datasets.

For evaluating the probe we apply the same affine transformation used to normalize the training set and project onto the probe direction to get a real-valued deception score at every token. We then take the mean across tokens to get a single deceptiveness-score for each model response. Thus, for the test_activation_dict we also need to save the information of which activations/tokens belong to which response.

Thus, for each model and each layer, we train 6 probes (3 on one dataset + 3 on combined datasets) each evaluated on the three full datasets test splits and on the extra/control datasets.
For each probe evaluation on a full dataset's test split we save: 
- the AUROC, 
- the recall at 1% false positive rate,
- the whole Receiver Operating Characteristic (ROC) curve (as a PNG),
- the probe scores on the honest and deceptive samples as two different distributions/histograms in the same plot (as a PNG), and
- a list containing the different probe scores together with the conversation id.
For the probe evaluation on the extra/control datasets we save:
- the probe scores on all samples as a distributions/histograms, and
- a list containing the different probe scores together with the conversation id.
"""

