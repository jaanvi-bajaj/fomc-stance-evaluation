# FOMC stance classification: which AI system should a bond desk use?

DS602 Topics in Data Science for Economics, mini-project for Lessons 5 and 6.
Jaanvi Bajaj and Nikhil Balaji, Master of Data Science in Economics, Singapore Management University, September 2026.

## What we did

The Federal Open Market Committee (FOMC) sets US interest rates, and every sentence it publishes can lean hawkish (towards higher rates), dovish (towards lower rates) or neutral. We imagined a bond desk that wants these sentences tagged automatically and asked which system it should use.

We compared a roberta-base model that we fine-tuned on 1,984 labelled sentences with Gemini 3.5 Flash-Lite and Gemini 3.5 Flash, both used zero-shot with the same prompt. We also included two baselines: one that always predicts neutral and one that guesses at random.

Not every mistake costs the same. If a model reads a hawkish sentence as dovish, the desk could trade in the wrong direction, so we tracked these flips separately and gave them a higher cost than other errors.

## Results on the 496-sentence test set

| System | Macro F1 (95% CI) | Flip rate | p95 latency | Cost per 1,000 sentences |
|---|---|---|---|---|
| RoBERTa (fine-tuned) | 0.660 (0.616 to 0.703) | 6.2% | 0.012 s | 571 |
| Gemini 3.5 Flash-Lite | 0.595 (0.545 to 0.643) | 2.8% | 0.961 s | 468 |
| Gemini 3.5 Flash | 0.622 (0.573 to 0.668) | 2.8% | 1.975 s | 462 |
| RoBERTa then Flash (cascade) | 0.676 | 2.8% | 1.741 s | 415 |

RoBERTa had the best macro F1 and was much faster, but it swapped hawkish and dovish more than twice as often as the Gemini models. Once those flips were priced, Gemini Flash and the cascade came out cheapest, and this held whether a flip cost 3, 5 or 10 times an ordinary error. None of the differences between systems was significant after a Holm correction. RoBERTa also failed on Hindi and Chinese translations and on negated sentences.

When we relabelled 50 test sentences ourselves without looking at the dataset labels, we agreed with them only 58% of the time (Cohen's kappa 0.31). Half of the 30 model errors we read by hand turned out to be sentences where the dataset label itself was debatable.

The cascade threshold was chosen on the test set, so its score is likely optimistic. Our recommendation is to use Gemini Flash for now and run the cascade alongside it until the threshold has been chosen on separate data and its latency is under one second.

## Files

- fomc_stance_eval.ipynb: the full evaluation notebook, run in Google Colab, with outputs
- DS602_FOMC_Stance_Evaluation_Report.pdf: our written report
- FOMC_Stance.pdf: presentation slides
- result_table.csv: all metrics for every system
- pairwise_tests.csv: bootstrap intervals, McNemar and Holm-adjusted p-values
- robustness_matrix.csv: macro F1 on each stress-test set
- figures: all charts from the notebook
- blind_50.csv: our blind relabelling of 50 test sentences
- negation_16.csv: 16 sentences rewritten with negation and relabelled
- fresh_fomc.csv: 14 sentences from the July and September 2026 FOMC statements
- error_buckets.csv: 30 model errors we read and categorised by hand

## Data

The main dataset is gtfintechlab/fomc_communication on Hugging Face (Shah, Paturi and Chava, ACL 2023), loaded directly in the notebook: https://huggingface.co/datasets/gtfintechlab/fomc_communication

The out-of-domain set is gtfintechlab/european_central_bank. The eight statements used for the stance index come from federalreserve.gov.

## Running it

Open the notebook in Colab with a T4 GPU and add a Gemini API key under Secrets as GEMINI_API_KEY. No key is stored in this repository. Put the four hand-labelled CSV files in MyDrive/fomc_eval/hand_labels and run all cells. The fine-tuned model is not included because it is too large for GitHub, so the notebook trains it again. Setting MOCK = True in the config cell runs the whole pipeline on fake data without any API calls.

## Limitations

Only one of us labelled the human-evaluation sets, and some stress-test sets are small (14 fresh and 16 negated sentences). The negated sentences were drafted with help from an AI assistant and checked by us. Flash-Lite ran with its thinking feature on while Flash ran with it off. We also used AI coding assistants for parts of the pipeline and reviewed every cell ourselves.
