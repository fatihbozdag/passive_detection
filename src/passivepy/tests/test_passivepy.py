import sys
import os
import pandas as pd
from termcolor import colored
import spacy

# Add the PassivePy source directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "PassivePyCode/PassivePySrc"))

try:
    from PassivePy import PassivePyAnalyzer
    print(colored("Successfully imported PassivePy", "green"))
except ImportError as e:
    print(colored(f"Error importing PassivePy: {e}", "red"))
    sys.exit(1)

def test_passivepy_with_examples():
    """Test PassivePy with known passive examples to debug why it's not detecting passives"""
    
    # Known passive examples that should be detected
    test_sentences = [
        "The book was written by John.",
        "The house is being built by the construction company.",
        "The document has been signed by all parties.",
        "The car was stolen yesterday.",
        "The problem is being looked into.",
        "The building was destroyed in the fire.",
        "The bill will be paid tomorrow.",
        "The decision was made without consulting us.",
        "The project is being completed on time.",
        "The data were collected over a six-month period."
    ]
    
    print(colored("Testing PassivePy with known passive examples:", "blue"))
    
    # Try with different spaCy models
    model_name = "en_core_web_sm"
    
    print(colored(f"\nTesting with model: {model_name}", "yellow"))
    try:
        # Load the spaCy model first
        try:
            nlp = spacy.load(model_name)
            print(colored(f"Successfully loaded spaCy model: {model_name}", "green"))
        except Exception as e:
            print(colored(f"Error loading spaCy model {model_name}: {e}", "red"))
            sys.exit(1)
            
        # Initialize PassivePy with loaded model
        analyzer = PassivePyAnalyzer(nlp=nlp)
        print(colored(f"Successfully initialized PassivePyAnalyzer with {model_name}", "green"))
        
        # Create DataFrame with test sentences
        df = pd.DataFrame({"text": test_sentences})
        
        # Test each sentence individually
        for i, sentence in enumerate(test_sentences):
            print(f"\nTesting sentence {i+1}: '{sentence}'")
            
            # Method 1: Use match_text function
            try:
                result = analyzer.match_text(sentence)
                is_passive = result["binary"].values[0] == 1
                print(f"Method 1 (match_text): {'PASSIVE' if is_passive else 'NOT PASSIVE'}")
                if is_passive:
                    print(f"  Passive matches: {result['all_passives'].values[0]}")
            except Exception as e:
                print(f"Error in match_text: {e}")
            
            # Method 2: Use match_corpus_level function
            try:
                result = analyzer.match_corpus_level(
                    pd.DataFrame({"text": [sentence]}),
                    column_name="text",
                    n_process=1,
                    batch_size=1
                )
                is_passive = result["binary"].values[0] == 1
                print(f"Method 2 (match_corpus_level): {'PASSIVE' if is_passive else 'NOT PASSIVE'}")
                if is_passive:
                    print(f"  Passive matches: {result['all_passives'].values[0]}")
            except Exception as e:
                print(f"Error in match_corpus_level: {e}")
        
        # Test all sentences together
        print(colored("\nTesting all sentences together:", "blue"))
        try:
            result = analyzer.match_corpus_level(df, column_name="text", n_process=1, batch_size=10)
            print(f"Detected {result['binary'].sum()} passive sentences out of {len(test_sentences)}")
            
            # Display results
            for i, (sentence, is_passive, matches) in enumerate(
                zip(test_sentences, result["binary"], result["all_passives"])
            ):
                status = "PASSIVE" if is_passive == 1 else "NOT PASSIVE"
                print(f"{i+1}. '{sentence}' -> {status}")
                if is_passive == 1:  # Fixed the deliberate error
                    print(f"  Matches: {matches}")
        except Exception as e:
            print(colored(f"Error in batch processing: {e}", "red"))
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(colored(f"Error initializing PassivePy: {e}", "red"))

if __name__ == "__main__":
    test_passivepy_with_examples() 