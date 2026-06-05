
This project is about white-box methods to detect deception in LLMs.
My personal holy-grail would be detecting unprompted deception, e.g., covering up insider trading, sandbagging, alignment faking -- without access to the CoT (so as to not rely on CoT faithfulness).

Related work
One paper covering a very similar problem is [Goldowsky-Dill] (Apollo) -- they train a linear probe that then detects deception in Llama-3.3-70B-Instruct in various, different scenarios.
However, as I showed in the last months and as [Liars' bench] also discusses, there exact training methodoly (mainly the choice of the training data set) applied to other models produces probes that perform very poorly.

What I did sofar
- Applied Apollo's methodology to another Llama model, Qwen models and Gemma models: the probes trained this way performed well on no other model (probes are trained on instructed-deception and tested on roleplaying)
- Investigated the difference between models generating text and prefilling models with reference text for probe training
- generated and collected activativations for 4 datasets for unprompted and system-promot-instructed deception 

What I'm doing now
- filtering the model answers on the 4 new datasets
- looking at other benchmarks

What I'm still planning to do in the near-term future
- training probes on the new datasets
- come up with a 'probe-failure taxonomy' -- if a probe fails to generalize to another dataset, what does this mean? where did the error come from? are the datasets too incompatible?

What I'm still planning to do in the mid-term future
- exploring natural-language-autoencoders to look for probe confounders
- also explore sycophancy as a form of deception and cross-test probes


