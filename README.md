

---

# nlp-robustness-comparison

This repository focuses on comparing the performance of three tools used for evaluating the robustness of NLP models in real-world conditions. Specifically, it evaluates the resistance of sentiment analysis models to noise and adversarial attacks using tools like **CheckList**, **TextAttack**, and **AugLy**.

### Goals:

* Evaluate the robustness of NLP models to noise and adversarial attacks.
* Compare the performance of three robustness testing tools: CheckList, TextAttack, and AugLy.
* Provide a reproducible testing pipeline for robustness evaluation.

## Repository Structure

The repository is organized as follows:

```
nlp-robustness-comparison/          
├── diagrams/                
├── results/                 
│   ├── results.csv          
│   ├── summary_by_test.csv  
│   ├── summary_for_tools.csv 
│   ├── table_augly.csv      
│   ├── table_checklist.csv  
│   ├── table_overall_tools_by_dataset.csv  
│   └── table_textattack.csv 
├── src/                     
│   ├── requirements.txt     
│   ├── nlp_test.py          
│   ├── diagrams.ipynb       
│   └── tables.ipynb         
├── notebooks/               
│   └── nlp_test.ipynb       
├── README.md                
```

### Description of Key Folders and Files:

* **diagrams/**: This folder holds multiple PNG images generated during the experiments, such as performance comparison graphs. The specific filenames may vary.
* **results/**: Stores the output results of the experiments, including:

  * **results.csv**: The main CSV file with aggregated experiment results.
  * **summary_by_test.csv**: Summary of results by test.
  * **summary_for_tools.csv**: Summary of results for each tool.
  * **table_augly.csv**, **table_checklist.csv**, **table_textattack.csv**: Results for each tool.
  * **table_overall_tools_by_dataset.csv**: Overall comparison of tools by dataset.
* **src/**: Contains the main scripts and notebooks.

  * **requirements.txt**: Lists the necessary Python libraries to run the experiment.
  * **nlp_test.py**: The script to execute the robustness evaluation of models.
  * **diagrams.ipynb**: A Jupyter notebook for generating and saving diagrams.
  * **tables.ipynb**: A Jupyter notebook for generating tables of experiment results.
* **notebooks/**: Contains the original Google Colab notebook for running the experiment.

  * **nlp_test.ipynb**: The Google Colab notebook for running the experiment interactively.
* **README.md**: This file contains the project description, setup instructions, and details on how to run the experiments.


## Datasets

The following datasets are used for evaluating the models:

1. **SST-2 (GLUE)**: A sentiment classification dataset.

   * Link: [SST-2](https://www.tensorflow.org/datasets/catalog/glue "glue - Datasets")
   * Description: A binary sentiment classification task, used as part of the GLUE benchmark.

2. **IMDb**: A sentiment classification dataset for movie reviews.

   * Link: [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb "stanfordnlp/imdb · Datasets at Hugging Face")
   * Description: A collection of movie reviews labeled as positive or negative.

3. **Emotion**: A multi-class dataset for emotion classification in text.

   * Link: [Emotion](https://huggingface.co/datasets/dair-ai/emotion "dair-ai/emotion · Datasets at Hugging Face")
   * Description: Contains text samples labeled with emotions such as happiness, sadness, anger, etc.

## Models

The following models are used for comparison:

1. **DistilBERT (SST-2)**: A smaller, faster version of BERT fine-tuned for sentiment analysis on SST-2.

   * Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
   * Link: [DistilBERT for SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

2. **DistilBERT (IMDb)**: A fine-tuned version of DistilBERT for sentiment analysis on the IMDb dataset.

   * Model: `lvwerra/distilbert-imdb`
   * Link: [DistilBERT for IMDb](https://huggingface.co/lvwerra/distilbert-imdb "lvwerra/distilbert-imdb")

3. **DistilBERT (Emotion)**: A fine-tuned version of DistilBERT for emotion classification.

   * Model: `transformersbook/distilbert-base-uncased-finetuned-emotion`
   * Link: [DistilBERT for Emotion](https://huggingface.co/transformersbook/distilbert-base-uncased-finetuned-emotion "transformersbook/distilbert-base-uncased-finetuned-emotion")

4. **Baseline Model**: **TF-IDF + Logistic Regression**, trained anew on each dataset.

## Tools

The experiment uses the following tools for robustness testing:

Вот исправленные ссылки для статей, используя актуальные данные:

1. **CheckList**: A tool for generating perturbations based on predefined linguistic transformations.

   * GitHub: [CheckList on GitHub](https://github.com/marcotcr/checklist)
   * Paper: [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://arxiv.org/abs/2005.04118)

2. **TextAttack**: A library for adversarial attack generation and defense.

   * GitHub: [TextAttack on GitHub](https://github.com/QData/TextAttack)
   * Paper: [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://aclanthology.org/2020.emnlp-demos.16/)

3. **AugLy**: A library for augmenting datasets with noise and transformations.

   * GitHub: [AugLy on GitHub](https://github.com/facebookresearch/AugLy)
   * Paper: [AugLy: Data Augmentations for Robustness](https://arxiv.org/abs/2201.06494)

---


Each tool is used to generate test cases with different types of modifications, including:

* **Noise**: Random modifications such as punctuation changes and character-level noise.
* **Adversarial Attacks**: Intentional manipulations designed to confuse models.


## How to Run the Experiments

### For Script-Based Execution:

1. Clone the repository:

   ```bash
   git clone https://github.com/rwlf-i/nlp-robustness-comparison.git
   cd nlp-robustness-comparison
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the experiment:

   ```bash
   python src/nlp_test.py
   ```

4. To generate diagrams, execute the following command:

   ```bash
   jupyter nbconvert --to notebook --execute src/diagrams.ipynb
   ```

5. To generate tables with experiment results:

   ```bash
   jupyter nbconvert --to notebook --execute src/tables.ipynb
   ```

### For Google Colab Execution:

To run the experiment in Google Colab, simply open the notebook `nlp_test.ipynb` in the `notebooks/` folder. You can download the notebook from the repository, upload it to your own Google Colab, and run the cells sequentially. Each cell contains the same commands, but with `!` prefix for shell commands in Colab.

**Example for running in Colab:**

```python
!git clone https://github.com/rwlf-i/nlp-robustness-comparison.git
%cd nlp-robustness-comparison
!pip install -r requirements.txt
!python3 src/nlp_test.py
!jupyter nbconvert --to notebook --execute src/diagrams.ipynb
!jupyter nbconvert --to notebook --execute src/tables.ipynb
```

### Notes:

* Ensure you have Jupyter installed for executing the notebooks locally.
* The paths in the notebooks assume you are running them from the root of the repository.
* For Colab execution, use the provided commands in each cell with the `!` prefix for proper execution.


## Evaluation Metrics

In the `results.csv` file, the following metrics are recorded:

In the experiment, the following metrics are recorded in the `result.csv` file:

1. **run_id**, **seed**, **repeat_id**: Identifiers for the specific run, the random seed used, and the repeat number for the experiment. This allows for reproducibility of results.

2. **dataset_key**, **dataset_hf_id**, **hf_model_name**: These fields store information about the dataset and model used in the experiment. The `hf_model_name` indicates the pre-trained model name (e.g., DistilBERT fine-tuned for SST-2).

3. **n_labels**: The number of unique labels in the dataset, used to determine the classification task (binary or multi-class).

4. **f1_avg**: The average F1 score across all labels. This metric balances precision and recall, providing a single value to summarize model performance.

5. **dataset**: The name of the dataset used for the experiment.

6. **eval_size**: The number of evaluation samples used in the experiment.

7. **model**: The model used for testing, such as TF-IDF + Logistic Regression (`sk_tfidf_lr`) or DistilBERT (`hf_distilbert`).

8. **tool**: The tool used for generating perturbations (e.g., "clean", "augly", "checklist", "textattack").

9. **test_id**: Identifies the specific test or perturbation applied.

10. **n_eval**: The total number of evaluations conducted.

11. **n_ok**: The number of test cases where the model produced a valid output.

12. **fail_rate**: The proportion of failed predictions, calculated as `(1 - n_ok / n_eval)`.

13. **changed_rate**: The proportion of test cases where the prediction changed after applying a perturbation.

14. **avg_similarity**: The average similarity between the original and modified text, computed using the `SequenceMatcher` (from the `difflib` library). This metric reflects how much the perturbation has altered the original text.

15. **invariance_rate**: The proportion of test cases where the prediction remained the same after applying the perturbation (i.e., the model's resistance to changes).

16. **acc**: Accuracy of the model's predictions after applying the perturbation.

17. **f1**: F1 score after applying the perturbation, balancing precision and recall.

18. **acc_clean** and **f1_clean**: Accuracy and F1 score on the clean (unperturbed) data, used for comparison with post-perturbation results.

19. **drop_acc** and **drop_f1**: The drop in accuracy and F1 score compared to the clean data, calculated as `acc_clean - acc` and `f1_clean - f1`.

20. **time_sec**: The time taken to process the evaluation.

21. **peak_rss_mb**: The peak memory usage (in MB) during the evaluation.

22. **attack_success_rate**: The success rate of adversarial attacks, measured as the proportion of successful attacks.

23. **notes**: Additional information about the experiment, such as the configuration of the attack or other relevant details.


These metrics help assess the model's resilience to noise and adversarial attacks, and provide a comprehensive picture of its performance across different conditions.



---

## Results

The experiment produces the following outputs:

* **Tables**: Comparison of models' performance on the original data and after applying noise or attacks. 
* **Diagrams**: Visualizations of model performance before and after perturbations, illustrating the impact of different tools.

The results presented in Tables 1 and 2 are aggregated, showing the mean values and standard deviations in a single cell for clarity.

---

### Table 1 – Results for AugLy and CheckList

| tool      | test_id                 | model      | drop_acc    | drop_f1     | time_mean   | avg_similarity | invariance_rate |
| --------- | ----------------------- | ---------- | ----------- | ----------- | ----------- | -------------- | --------------- |
| AugLy     | augly_insert_punct      | DistilBERT | 0.46 ± 0.09 | 0.7 ± 0.21  | 3.35 ± 0.62 | 0.28 ± 0.12    | 0.44 ± 0.08     |
| AugLy     | augly_insert_punct      | TF-IDF_LR  | 0.38 ± 0.06 | 0.55 ± 0.32 | 0.08 ± 0.07 | 0.28 ± 0.12    | 0.5 ± 0.04      |
| AugLy     | augly_insert_whitespace | DistilBERT | 0.29 ± 0.06 | 0.34 ± 0.1  | 2.45 ± 0.86 | 0.31 ± 0.13    | 0.65 ± 0.06     |
| AugLy     | augly_insert_whitespace | TF-IDF_LR  | 0.38 ± 0.06 | 0.55 ± 0.32 | 0.08 ± 0.1  | 0.31 ± 0.13    | 0.5 ± 0.04      |
| AugLy     | augly_insert_zwsp       | DistilBERT | 0.0 ± 0.0   | 0.0 ± 0.0   | 1.48 ± 1.28 | 0.32 ± 0.14    | 1.0 ± 0.0       |
| AugLy     | augly_insert_zwsp       | TF-IDF_LR  | 0.38 ± 0.06 | 0.55 ± 0.32 | 0.09 ± 0.09 | 0.32 ± 0.14    | 0.5 ± 0.04      |
| AugLy     | augly_unicode           | DistilBERT | 0.12 ± 0.06 | 0.14 ± 0.05 | 1.73 ± 1.5  | 0.69 ± 0.22    | 0.82 ± 0.05     |
| AugLy     | augly_unicode           | TF-IDF_LR  | 0.07 ± 0.04 | 0.07 ± 0.04 | 0.37 ± 0.38 | 0.69 ± 0.22    | 0.84 ± 0.07     |
| CheckList | checklist_typos         | DistilBERT | 0.03 ± 0.03 | 0.03 ± 0.03 | 1.37 ± 1.19 | 0.99 ± 0.01    | 0.95 ± 0.04     |
| CheckList | checklist_typos         | TF-IDF_LR  | 0.02 ± 0.03 | 0.02 ± 0.03 | 0.07 ± 0.07 | 0.99 ± 0.01    | 0.95 ± 0.04     |

---

### Table 2 – Results for TextAttack

| tool       | test_id               | model      | n_eval | fail_rate   | drop_acc    | drop_f1     | time_mean     | asr_mean   | avg_similarity | changed_rate |
| ---------- | --------------------- | ---------- | ------ | ----------- | ----------- | ----------- | ------------- | ---------- | -------------- | ------------ |
| TextAttack | textattack_textfooler | DistilBERT | 10     | 0.1 ± 0.11  | 0.76 ± 0.31 | 0.75 ± 0.32 | 18.16 ± 12.28 | 0.77 ± 0.3 | 0.92 ± 0.05    | 0.77 ± 0.3   |
| TextAttack | textattack_textfooler | TF-IDF_LR  | 40     | 0.16 ± 0.06 | 0.7 ± 0.35  | 0.71 ± 0.35 | 81.03 ± 34.65 | 0.7 ± 0.35 | 0.91 ± 0.05    | 0.7 ± 0.35   |


---

### Interpretation of Results:

* **AugLy** demonstrated the lowest **drop in accuracy** and **drop in F1 score**, indicating that the model's performance was more resilient to the transformations applied by this tool. However, the **time** taken for these transformations was relatively low compared to **TextAttack**.
* **TextAttack** showed a higher **drop in accuracy** and **drop in F1**, indicating that the adversarial attacks were more effective in reducing model performance. Additionally, **TextAttack** required significantly more time to generate and apply attacks, especially for larger models such as **DistilBERT**.
* **CheckList** exhibited the smallest **drop in accuracy** and **drop in F1**, suggesting that linguistic perturbations (e.g., typos, syntactic changes) had a smaller impact on model performance compared to the adversarial attacks generated by **TextAttack**.

### Conclusion:

The results highlight the trade-offs between **AugLy**, **TextAttack**, and **CheckList** regarding their impact on model robustness. **TextAttack** provides the most aggressive perturbations but requires significantly more time, while **CheckList** offers a lightweight method with minimal impact on performance. **AugLy** falls somewhere in between, offering moderate impact with lower time overhead.

---

### Author Information

**Main author**  

Malova Anastasia Sergeevna

**Advisor and minor author**  

Parkhomenko Vladimir Andreevich

**Peter the Great St. Petersburg Polytechnic University**  

Course: Software Testing Methods

---



