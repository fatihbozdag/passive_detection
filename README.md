# PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data

**_This repository is the code of the following paper, so if you use this package, please cite the work as:
Sepehri, A., Mirshafiee, M. S., & Markowitz, D. M. (2022). PassivePy: A tool to automatically identify passive voice in big text data. Journal of Consumer Psychology. [Preprint available](https://doi.org/10.1002/jcpy.1377)**


Our aim with this work is to create a reliable (e.g., passive voice judgments are consistent), valid (e.g., passive voice judgments are accurate), flexible (e.g., texts can be assessed at different units of analysis), replicable (e.g., the approach can be performed by a range of research teams with varying levels of computational expertise), and scalable way (e.g., small and large collections of texts can be analyzed) to capture passive voice from different corpora for social and psychological evaluations of text. To achieve these aims, we introduce PassivePy, a fully transparent and documented Python library.

For accessing the datasets in our paper, please click on [this link](https://osf.io/j2b6u/?view_only=0e78d7f4028041b693d6b64547b514ca). 

If you haven't used Python before or need detailed instructions about how to use this package please visit [our website](https://passivepy.streamlit.app/).


First we have to install the requirements in the following way (all requirements are needed for spaCy or other libraries we use.):
```
!pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements_lg.txt
!pip install PassivePy==0.2.21

```
Then, import PassivePy and initiate the analyzer:

```
from PassivePySrc import PassivePy

passivepy = PassivePy.PassivePyAnalyzer(spacy_model = "en_core_web_lg")
```
Use passivepy for single sentences:
```
# Try changing the sentence below:
sample_text = "The painting has been drawn."
passivepy.match_text(sample_text, full_passive=True, truncated_passive=True)
```
The output will be:
```
sentence : the input sentence
binary : Whether any passive voice is detected 
passive_match(es) : The span of passive form in text
raw_passive_count : Number of passive voices
```
You can set the full_passive or truncated_passive to true if you want the same sort of output for these two types of passive. (truncated is a passive without an object of preposition, while a full passive is one with the object of preposition - e.g., this was broken by him.)


For processing datasets, we have can either analyze the records sentence- or corpus-level. Your dataset can be in any format (e.g., CSV, XLSX or XLS).; however, make sure to that it has at least one column with the text that needs to be analyzed.

In case of large datasets, you can also add `batch_size = ...` and `n_process=...` to speed up the analysis (the default for both is 1).


``` 
# sentence level:
df_detected_s = passivepy.match_sentence_level(df, column_name='documents', n_process = 1,
                                                batch_size = 1000, add_other_columns=True,
                                                truncated_passive=False, full_passive=False)

# corpus level
df_detected_c = passivepy.match_corpus_level(df, column_name='sentences', n_process = 1,
                                            batch_size = 1000, add_other_columns=True,
                                            truncated_passive=False, full_passive=False)
```
In the output you will have a data frame with the following columns:

```
# corpus level
document : Records in the input data frame
binary : Whether a passive was detected in that document
passive_match(es) : Parts of the document detected as passive
raw_passive_count : Number of passive voices detected in the sentence
raw_passive_sents_count : Number of sentences with passive voice
raw_sentence_count : Number of sentences detected in the document
passive_sents_percentage : Proportion of passive sentences to total number of sentences

# Sentence level
docId : Initial index of the record in the input file
sentenceId : The ith sentence in one specific record
sentence : The detected sentence
binary : Whether a passive was detected in that sentence
passive_match(es) : The part of the record detected as passive voice
raw_passive_count : Number of passive forms detected in the sentence

```

If you needed to analyze each token of a sentence, i.g., print out the `DEP` (dependency), `POS` (coarse-grained part of speech tags), `TAG` (fine-grained part of speech tags), `LEMMA` (canonical form) of a word,  you can use the `parse_sentence` method of passivepy in the following way:

```
sample_sentence = "She has been killed"
passivepy.parse_sentence(sample_sentence)
```
And the output will be like the sample below:
```
word: She 
pos: PRON 
dependency: nsubjpass 
tag:  PRP 
lemma:  she
...
```



If you do not need any columns to be appended to the main dataset, simply add `add_other_columns = False`, or if you don't what the percentages to show up add `percentage_of_passive_sentences = False` in any of the following functions.


Accuracy on the CoLA dataset: 0.97
Accuracy on the CrowdSource Dataset: 0.98




# PassivePy Extension Project

This project extends the capabilities of PassivePy, a Python library for detecting passive voice in text. It includes a custom implementation that achieves comparable performance while offering enhanced features and improved accuracy in specific cases.

## Project Overview

The project focuses on improving passive voice detection through:

1. Advanced regex pattern matching
2. Dependency parsing with SpaCy
3. Special case handling for commonly missed passive constructions
4. Enhanced filtering to reduce false positives

## Repository Structure

- `my_passive_detector.py`: Our custom passive voice detection implementation
- `run_passive_detector.py`: Script for running PassivePy on text data
- `compare_implementations.py`: Script for comparing PassivePy and our custom implementation
- `analyze_icle_concordance.py`: Script for analyzing the ICLE concordance dataset
- `progress.md`: Documentation of project progress and findings
- `implementation_disagreements.csv`: Analysis of cases where implementations disagree
- `icle_annotated.csv`: ICLE dataset with passive voice annotations
- `icle_language_stats.csv`: Analysis of passive voice usage by native language

## Key Features

### Enhanced Passive Detection

Our implementation offers improved detection for:

- Election-related passives: "was elected", "was chosen", "was selected"
- Event-related passives: "was held", "was organized", "was cancelled"
- Negated passives: "isn't considered", "wasn't seen"
- "Left + participle" constructions: "was left saddened"
- Simple subject-verb passives: "Sports are played"
- By-agent passives across different tenses

### Reduced False Positives

Special filtering for:

- Adjectival states that look like passives: "is heated", "is excited"
- Emotional/mental states that aren't passive: "am stunned", "is surprised"
- Active voice constructions that resemble passives: "is having"
- Non-passive copular constructions: "is topic", "is subject"

## Performance

Our custom implementation:

- Achieves the same F1-score as PassivePy (0.85) on the crowdsource dataset
- Provides better recall (0.84 vs 0.77 for PassivePy)
- Shows strong agreement with human annotations (Cohen's Kappa: 0.8186)

## ICLE Analysis Findings

Analysis of 10,000 sentences from the ICLE concordance dataset revealed:

- 98.0% of sentences contain passive voice constructions
- Average of 3.16 passive phrases per sentence
- 12.2% of passive sentences contain by-agents
- Chinese-Mandarin, Russian, and Dutch language groups show high rates of passive use

## Getting Started

### Prerequisites

- Python 3.6+
- SpaCy with English model: `pip install spacy && python -m spacy download en_core_web_sm`
- Required libraries: pandas, numpy, matplotlib, seaborn, tqdm

### Usage

1. **Analyze text with custom detector:**

```python
import my_passive_detector as detector
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The bill was approved by the committee."
result = detector.process_text(text, nlp)
print(f"Is passive: {result['is_passive']}")
print(f"Passive phrases: {result['passive_phrases']}")
```

2. **Analyze ICLE dataset:**

```bash
python analyze_icle_concordance.py
```

3. **Compare implementations:**

```bash
python compare_implementations.py
```

## Contributing

Contributions are welcome! Areas for improvement include:

- Enhanced detection of complex passive constructions
- Better handling of domain-specific passive forms
- Language-specific passive detection customizations

## License

This project is available under the MIT License.

## Acknowledgments

- Original PassivePy library
- ICLE (International Corpus of Learner English) project
- SpaCy for providing excellent natural language processing capabilities



