import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
from termcolor import colored
import time
import regex as re
from itertools import chain 
import string
from tqdm import tqdm 
import tqdm.notebook as tq
import os, sys


try: 
    from PassivePyCode.PassivePySrc.PassivePyRules_FullPassive import create_matcher_full
    from PassivePyCode.PassivePySrc.PassivePyRules_TruncatedPassive import create_matcher_truncated
except: 
     
    from PassivePySrc.PassivePyRules_FullPassive import create_matcher_full
    from PassivePySrc.PassivePyRules_TruncatedPassive import create_matcher_truncated

class PassivePyAnalyzer:
    
        """
            Get the data from a dataframe.

            Clean the dataset based on the given regex patterns.
            Match passive voice sentence level or corpus level.
            save the output to a file

        """
        def __init__(self, spacy_model = "en_core_web_lg"):

            """
            Create the Detector

            n_processses: number of core to use
            batch_size: size of batches of records passed onto the matcher
            regex_patterns: Patterns that should be detected and cleaned from the data
            
            
            """
            # os.system('pip install -r https://raw.githubusercontent.com/mitramir55/PassivePy/main/PassivePyCode/PassivePySrc/requirements.txt')
            self.nlp, self.matcher_t = create_matcher_truncated(spacy_model)
            self.nlp, self.matcher_f = create_matcher_full(spacy_model)


        def print_matches(self, sentence, matches):
            """
            prints match span
            """
            doc = self.nlp(sentence)
            if matches:
                for id_, s,e in matches:
                    match_ = doc[s:e] 
                    print(match_)
                    print(colored(self.nlp.vocab.strings[id_], 'blue'))



        def parse_sentence(self, sentence, truncated_passive=False):
            """
            This function allows us to see the components of a sentence, 
            specifically, the POS, DEP, and lemma
            """
            doc = self.nlp(sentence)
            
            for token in doc:
                print('word:', colored(token.text, 'green'), '\npos:', token.pos_,
                    '\ndependency:', token.dep_, '\ntag: ', token.tag_,
                    '\nlemma: ', token.lemma_)

            full_passive_matches = self.matcher_f(doc)
            if truncated_passive: truncated_passive_matches = self.matcher_t(doc)

            self.print_matches(sentence, full_passive_matches)
            self.print_matches(sentence, truncated_passive_matches)



        def _detect_sents(self, cleaned_corpus, batch_size, n_process):

            print('Detecting Sentences...')

            """
            Separates sentences from each other in each record
             and puts them in a list along side the count of sentences in each 
             document in another list
             """
            cleaned_corpus = [corpus.lower() for corpus in cleaned_corpus]

            all_sentences = []
            count_sents = []
            unwanted = []

            # go through all the records
            m = 0
            for record_doc in tq.tqdm(self.nlp.pipe(cleaned_corpus, batch_size=batch_size, n_process = n_process), 
                                    leave=True,
                                    position=0,
                                    total=len(cleaned_corpus)):


                sentences = list(record_doc.sents)
                sentences = [str(sentence) if len(sentence)>=2 else 'Not a Sentence' for sentence in sentences] 


                for sentence in sentences:
                    i = sentences.index(sentence)
                
                    
                    #...........................joining with the previous one.............................#
                    # ones that start with but and their previous record doesn't have dot at its end
                    if i!=0:
                        if (re.search(r'^ *but', sentence) and not re.search(r'.$', sentences[i-1])) or all((re.search(r'^[A-Z0-9]', word) or re.search(r'^[\(\)\.\-]', word)) for word in sentence.split()) or re.search(r'^\(.*\)[\.\!\,]*', sentence):
                            j = 0
                            for j in range(1, i):
                                if i-j not in unwanted:
                                    sentences[i-j] = sentences[i-j] + sentences[i]
                                    unwanted.append(i)
                                    break
                            

                    #.........................joining with the next one..........................#
                    if i != len(sentences)-1:


                        if re.search(r', *$', sentence): # remove the one that's ended with comma
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        if re.search(r'\- *$', sentence): 
                            # see if it's ended with hyphen then look at the next one
                            # if it has and in the beginning, forget about this one and go to the next to analyze the and 
                            # and not duplicate the process
                            if re.search(r'^ *(\([\w\. ]*\))* *and', sentences[i+1]):
                                continue
                            else: 
                                # but if there was no and in the next one,
                                #  join this with the next

                                sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                                unwanted.append(i+1)
                        # see if it ends with and and join it with the 
                        elif re.search(r'and *$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        # end with 'as well as' and join with the next one
                        elif re.search(r'((as well as) *)$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)

                        # end with the following phrases and join with the next ones
                        elif re.search(r'((Exp\.)|(e\.g\.)|(i\.e\.))$', sentence):
                            sentences[i] = ' '.join([sentences[i], sentences[i+1]])
                            unwanted.append(i+1)


                m+=1
                for index in sorted(set(unwanted), reverse=True):
                    del sentences[index]
                unwanted = []

                
                count_sents.append(len(sentences))
                all_sentences.append(sentences) 

            all_sentences = list(chain.from_iterable(all_sentences))
            print(f'Total number of sentences = {len(all_sentences)}')


            return np.array(count_sents, dtype='object'), np.array(all_sentences, dtype='object')


        def _find_doc_idx(self, count_sents):

            """ finds the indices required for the documents and sentences"""

            m = 1
            sent_indices = []
            doc_indices = []
            for i in count_sents:
                n = 1
                for j in range(i):
                    sent_indices.append(n)
                    doc_indices.append(m)
                    n+=1
                m+=1
            return pd.DataFrame(sent_indices), pd.DataFrame(doc_indices)


        def _add_other_cols(self, df, column_name, count_sents):

            """ creates a dataframe of all the other columns
            with the required number of repetitions for each """

            # create a list of all the col names
            fields = df.columns.tolist()
            # remove column_name
            del fields[fields.index(column_name)]

            other_columns = {}
            # create a df of all the other cols with 
            # appropriate number of repetitions
            for col in fields:
                properties = []
                for i in range(len(count_sents)):
                    properties.append(count_sents[i]*[df.loc[i, col]])
                
                properties = list(chain.from_iterable(properties))
                other_columns[col] = properties

            df_other_cols = pd.DataFrame.from_dict(other_columns)

            return df_other_cols  


        def match_text(self, cleaned_corpus, batch_size=1, n_process=1, , truncated_passive=False):

            """ 
            This function finds passive matches in one sample sentence
            """

            # we don't want to print the usual statements
            with HiddenPrints():
                # seperating sentences
                count_sents, all_sentences = self._detect_sents([cleaned_corpus], batch_size, n_process)
                output_df = self._find_matches(all_sentences, batch_size, n_process, truncated_passive)
                s_output = pd.DataFrame(output_df)
                

                return s_output

                
        def _find_unique_spans(self, doc, truncated_passive=False) ->list:

            """"
            finds matches and checks for overlaps
            """

            final_matches_i = []
            if truncated_passive: matches_i = self.matcher_t(doc)
            else: matches_i = self.matcher_t(doc)

            if matches_i:
                spans = [doc[s:e] for id_, s,e in matches_i]

                for span in spacy.util.filter_spans(spans):
                    final_matches_i.append(str(span))
            return final_matches_i


        def _find_matches(self, sentences, batch_size, n_process, truncated_passive=False) -> dict:

            """ finds matches from each record """
            print(colored('Starting to find passives...', 'green'))  

            # full passive parameters
            raw_full_passive_count = []
            full_passive_matches = []
            binary_full_passive = []

            # truncated passive parameters
            raw_truncated_passive_count = []
            truncated_passive_matches = []
            binary_truncated_passive = []

            i = 0

            
            for doc in tq.tqdm(self.nlp.pipe(sentences, batch_size=batch_size, n_process=n_process), 
                                    leave=True,
                                    position=0,
                                    total=len(sentences)):

                binary_f = 0
                binary_t = 0
                
                # truncated passive voice ----------------------------------
                if truncated_passive:

                    truncated_matches_i = self._find_unique_spans(doc, truncated_passive)
                    if truncated_matches_i != []:
                        binary_t = 1
                        binary_truncated_passive.append(binary_t)
                        truncated_passive_matches.append(truncated_matches_i)
                        raw_truncated_passive_count.append(len(truncated_matches_i))

                    # if there were no matches
                    else:
                        truncated_passive_matches.append(None)
                        raw_truncated_passive_count.append(0)
                        binary_truncated_passive.append(binary_t)

                # full passive voice ----------------------------------------
                full_matches_i = self._find_unique_spans(doc, truncated_passive=False)
                if full_matches_i != []:

                    binary_f = 1
                    full_passive_matches.append(full_matches_i)
                    raw_full_passive_count.append(len(full_matches_i))
                    binary_full_passive.append(binary_f)

                # if there were no matches
                else:
                    full_passive_matches.append(None)
                    raw_full_passive_count.append(0)
                    binary_full_passive.append(binary_f)
                    

                i+=1
            output_dict = {}

            columns = [sentences, full_passive_matches, raw_full_passive_count,
             binary_full_passive, binary_full_passive]

            if truncated_passive: 
                columns += [truncated_passive_matches, raw_truncated_passive_count,
                 binary_truncated_passive]

            for element in columns:
                # name of variables will be the name of columns 
                element_name = [ k for k,v in locals().items() if v is element][0]
                output_dict[str(element_name)] = np.array(element, dtype='object')
            
            return output_dict


        def match_sentence_level(self, df, column_name, n_process = 1,
                                batch_size = 1000, add_other_columns=True,
                                truncated_passive=False):

            """
            Parameters

            column_name: name of the column with text
            level: whether the user wants corpus level or sentence level results
            n_process: number of cores to use can be any number
            between 1 and the maximum number of cores available
            (set it to -1 to use all the cores available)
            batch_size: give records in batches to the matcher
            record when passed
            add_other_columns: True\False whether or not to add the other columns 
            to the outputted dataframe
            """
            
            df = df.reset_index(drop=True)
            # create a list of the column we will process
            cleaned_corpus = df.loc[:, column_name].values.tolist()

            # seperating sentences
            count_sents, all_sentences = self._detect_sents(cleaned_corpus, batch_size, n_process)

            # find indices required for the final dataset based on the document and sentence index
            sent_indices, doc_indices = self._find_doc_idx(count_sents)

            # create a df of matches -------------------------------------------
            output_dict = self._find_matches(all_sentences, batch_size, n_process, truncated_passive)
            s_output = pd.DataFrame(output_dict)
            
            # add indices
            s_output.insert(0, "docId", doc_indices)
            s_output.insert(1, "sentenceId", sent_indices)


            # concatenating the results with the initial df -------------------
            if add_other_columns==True:

                other_cols_df = self._add_other_cols(df, column_name, count_sents)
                assert len(other_cols_df) == len(s_output)
                df_final = pd.concat([s_output, other_cols_df], axis = 1)

                return df_final

            else:
                return s_output


        def _all_elements_in_one_list(series_: pd.Series(list)) -> list:
            """
            a function for reducing the size of a series
            """
            # output: 1d list
            passive_matches = [val for val in series_.values if val!=None]
            passive_matches = list(chain.from_iterable(passive_matches))
            return passive_matches


        def match_corpus_level(self, df, column_name, n_process = 1,
            batch_size = 1000, add_other_columns=True, truncated_passive=False):

            """
            finds matches based on sentences in all records

            Parameters

            column_name: name of the column with text
            level: whether the user wants corpus level or sentence level
            results
            n_process: number of cores to use can be any number
            between 1 and the maximum number of cores available
            (set it to -1 to use all the cores available)
            batch_size: give records in batches to the matcher
            record when passed
            add_other_columns: True\False whether or not to add the other columns 
            to the outputted dataframe
            sentences to the output dataset
            """
            
            df = df.reset_index(drop=True)
            # create a list of the column we will process
            cleaned_corpus = df.loc[:, column_name].values.tolist()


            s_output = self.match_sentence_level(df, column_name, n_process = n_process,
                            batch_size = batch_size, add_other_columns=add_other_columns,
                            truncated_passive=truncated_passive)

            # define lists for all values--------------------
            # full passive
            full_passive_matches = []
            full_passive_count = []
            full_passive_binary = []
            full_passive_percentages = []
            full_passive_sents_count = []

            # truncated
            truncated_passive_matches = []
            truncated_passive_count = []
            truncated_passive_binary = []
            truncated_passive_percentages = []
            truncated_passive_sents_count = []
            
            # general
            count_sents = []

            # list all the docs
            ids_ = s_output.docId.unique()


            for i in tq.tqdm(ids_, leave=True, position=0, total=len(ids_)):

                # select all the sentences of a doc
                rows = s_output[s_output['docId'] == i]

                # concatenate all the proberties ------------------------------------
                count_sents.append(len(rows))

                # full passive
                count_full_passive_s = sum(rows.binary_full_passive)
                percent_full =  count_full_passive_s/ len(rows)
                full_passive_sents_count.append(count_full_passive_s)
                full_passive_percentages.append(percent_full)

                # binary will be =1 if there is even one 1 
                if any(rows.binary_full_passive) == 1: full_passive_binary.append(1)
                else: full_passive_binary.append(0)

                # truncated passive
                if truncated_passive:
                    count_truncated_passive_s = sum(rows.binary_truncated_passive)
                    percent_truncated = count_truncated_passive_s/ len(rows)
                    truncated_passive_sents_count.append(count_truncated_passive_s)
                    truncated_passive_percentages.append(percent_truncated)

                    # binary will be =1 if there is even one 1 
                    if any(rows.binary_full_passive) == 1: truncated_passive_binary.append(1)
                    else: truncated_passive_binary.append(0)




                # putting all sentences' passives in one list ----------------------------
                # full passive
                full_passives = self._all_elements_in_one_list(rows['full_passive_matches'])
                full_passive_matches.append(full_passives)
                full_passive_count.append(len(full_passives))

                # truncated passive
                if truncated_passive:
                    truncated_passives = self._all_elements_in_one_list(rows['truncated_passive_matches'])
                    truncated_passive_matches.append(truncated_passives)
                    truncated_passive_count.append(len(truncated_passives))

            # put all properties in a dict ------------------------------------------------------
            output_dict = {}
            all_columns = [
                cleaned_corpus, count_sents, full_passive_matches, full_passive_count, 
                full_passive_sents_count, full_passive_percentages, full_passive_binary
                ]

            if truncated_passive: all_columns += [
                truncated_passive_matches, truncated_passive_count, truncated_passive_sents_count,
                truncated_passive_percentages, truncated_passive_binary
                ]
                
            for element in all_columns:
                output_dict[str(element)] = np.array(element, dtype='object')


            
            assert len(cleaned_corpus) == len(count_sents) == len(full_passive_count) == len(full_passive_matches) == len(full_passives_c) == len(count_p_sents)
            d_output = pd.DataFrame(output_dict)
               

            if add_other_columns==True:

                # create a list of all the col names
                fields = df.columns.tolist()
                # remove column_name
                del fields[fields.index(column_name)]

                assert len(df[fields]) == len(d_output)

                d_output = pd.concat([d_output, df[fields]], axis = 1)
                

            
            return d_output


# for stopping the print statements in one sample sentences
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
