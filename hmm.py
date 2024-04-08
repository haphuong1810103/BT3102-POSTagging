# Implement the four functions below

#Question 2.1a

def MLE_predict(training_file):
    #create two dictionary to store the count of tags and the count of pair tags and words
    # read the training_file and split the words and tags 
    # for each word and tag pair, increment the count of the tag and the pair tag and word
    
    tag_count = {}
    pair_count = {}

    with open(training_file, 'r') as file:
        for line in file:
            line_elements = line.strip()
            if len(line_elements) >= 2:
                token, tag = line_elements.split()
                if tag not in tag_count:
                    tag_count[tag] = 0
                    
                tag_count[tag] += 1

                if token not in pair_count:
                    pair_count[token] = {}
                if tag not in pair_count[token]:
                    pair_count[token][tag] = 0
                
                pair_count[token][tag] += 1


    #Dictionary of dictionary to store the output probabilties
    #key: token. value: {tag: P(token|tag)=count(token, tag)/count(tag)}
    #smoothing value: 0.01 
    delta = 0.1
    result = {}
    for token in pair_count:
        result[token] = {}
        for tag in pair_count[token]:
            result[token][tag] = (pair_count[token][tag] + delta) / (tag_count[tag] + delta * (len(pair_count) + 1))
    
    with open('naive_output_probs.txt', 'w') as outfile:
        for token in result:
            for tag in result[token]:
                outfile.write(f'{token} {tag} {result[token][tag]}\n')
MLE_predict('twitter_train.txt')

#Question 2.1b
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    output_probs = {}
    tag_count = {}
    with open(in_output_probs_filename, 'r') as file:
        for line in file:
            token, tag, prob = line.strip().split()
            if token not in output_probs:
                output_probs[token] = {}
            output_probs[token][tag] = float(prob)

            if tag not in tag_count:
                tag_count[tag] = 0
            tag_count[tag] += 1

    tweets = []
    with open(in_test_filename, 'r') as file:
        for line in file: 
            line = line.strip('\n')
            if line:
                tweets.append(line)

    predicted_tags = []
    for token in tweets:
        if token in output_probs:
            predicted_tag = max(output_probs[token].keys(), key=output_probs[token].get)
        else:
            # if token unseen, predict its tag as N 
            predicted_tag = 'N' #default tag is Noun 
        predicted_tags.append(predicted_tag)
    
    with open(out_prediction_filename, 'w', encoding="utf-8") as outfile:
        for tag in predicted_tags:
            outfile.write(f'{tag}\n')

# Question 2.1c
#Naive prediction accuracy:     1004/1378 = 0.7285921625544267


# Question 2.2a
'''
By Bayes' rule: 
    P(y = j | x = w) = [ P(x = w | y = j) * P(y = j) ] / P(x = w)

    where:
        P(y = j) = count(j) / count(total tags)
        P(x = w) = count(w) / count(total tokens)
        P(x = w | y = j) = count(w,j) / count(j).
    
    Thus we can simplify the formula to:
        [p(x = w | y = j) * count(j) / count(total tags) ] / [count(w) / count(total tokens)]
        
    Since each token is assigned to a single tag, count(total tags) == count(total tokens). 
    Thus we can simplify the formula to: 
        [p(x = w | y = j) ] * [ count(j) / count(w)]

    We can get P(x = w | y = j) from the naive_output_probs file
    We can get count(j) / count(w) by counting the number of times a tag appears in the training data and dividing it by the total number of tags/tokens
'''

# Question 2.2b
def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    
    #count the number of tags and the number of each tag
    tag_count = {}
    number_of_tags = 0
    with open(in_train_filename, 'r') as file:
        for line in file: 
            line_elements = line.strip()
            if len(line_elements) >= 2:
                number_of_tags += 1
                token, tag = line_elements.split()
                if tag in tag_count:
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1

    # change Naive2_probs to P(x = w|y = j)P(y = j) from naive output probs.txt.
    naive2_probs = {}
    with open(in_output_probs_filename, 'r') as file:
        for line in file:
            token, tag, prob = line.strip().split()
            if token not in naive2_probs:
                naive2_probs[token] = {}
            naive2_probs[token][tag] = float(prob)*tag_count[tag] / number_of_tags
    
    #store the result
    tweets = []
    with open(in_test_filename, 'r') as file:
        for line in file: 
            line = line.strip('\n')
            if line:
                tweets.append(line)

    predicted_tags2 = []
    for token in tweets:
        if token in naive2_probs:
            predicted_tag = max(naive2_probs[token], key=naive2_probs[token].get)
        else:
            # if token unseen, predict its tag as N 
            predicted_tag = 'N' #default tag is Noun 
        predicted_tags2.append(predicted_tag)

    with open(out_prediction_filename, 'w', encoding="utf-8") as outfile:
        for tag in predicted_tags2:
            outfile.write(f'{tag}\n')

# Question 2.2c
# Naive prediction2 accuracy:    1016/1378 = 0.737300435413643


''' 
Question 3
'''
#(a)

# define the function to calculate the transition probabilities, including the "start" and "stop" states
def trans_probs(in_train_filename, in_tags_filename, out_trans_probs_filename):
    tag_tag_counts = {}
    tag_counts = {}

    hidden_states = []
    with open(in_tags_filename, 'r') as tags: 
        for line in tags:
            hidden_states.append(line.strip())
    hidden_states.append("stop")

    # get the transition probabilities: run the whole train file to count and counts how many times a tag appears after another tag
    with open(in_train_filename, 'r') as train:
        prev_tag = "start"
        for line in train:
            tokens = line.split()  
            if len(tokens) < 2:
                if prev_tag in tag_tag_counts:
                    if "stop" in tag_tag_counts[prev_tag]:
                        tag_tag_counts[prev_tag]["stop"] += 1
                    else:
                        tag_tag_counts[prev_tag]["stop"] = 1
                else:
                    tag_tag_counts[prev_tag] = {"stop": 1}
                prev_tag = "start"
            
            elif len(tokens) >= 2:
                tag = tokens[1]
                if prev_tag in tag_tag_counts:
                    if tag in tag_tag_counts[prev_tag]:
                        tag_tag_counts[prev_tag][tag] += 1
                    else:
                        tag_tag_counts[prev_tag][tag] = 1
                else:
                    tag_tag_counts[prev_tag] = {tag: 1}
                
                if prev_tag in tag_counts:
                    tag_counts[prev_tag] += 1
                else:
                    tag_counts[prev_tag] = 1
                prev_tag = tag
    
    for prev_tag in tag_tag_counts:
        for tag in hidden_states:
            if tag not in tag_tag_counts[prev_tag]:
                #for unseen transtition, count = 0
                tag_tag_counts[prev_tag][tag] = 0


    #change the counts to probabilities using the MLE formula with the smoothing value of 0.01
    with open(out_trans_probs_filename, "w") as output:
        for prev_tag, tag_dict in tag_tag_counts.items():
            for tag, count in tag_dict.items():
                #smoothing
                output.write(f'{prev_tag} {tag} {(count + 0.01)/(tag_counts[prev_tag] + 0.01 * (len(tag_counts)))}\n') #since there are no unseen tag, we excluded the "+1" in the denominator
                
trans_probs('twitter_train.txt', 'twitter_tags.txt', 'trans_probs.txt')

# define the function to calculate the emission probabilities       

def output_probs(in_train_filename, in_tags_filename):
    #get the hidden states
    hidden_states = []
    with open(in_tags_filename, 'r') as tags:
        hidden_states = [line.strip() for line in tags]
    
    # initialise the dictionary to store the emission counts
    emission_counts = {}
    for state in hidden_states:
        emission_counts[state] = {}

    # get emission counts: run the whole train file to count and counts how many times a token appears in a tag
    with open(in_train_filename, 'r') as train:
        for line in train:
            tokens = line.split()
            if len(tokens) >= 2:
                token = tokens[0]
                tag = tokens[1]
                if token.startswith('@USER'):
                    token = '@USER'
                
                if token not in emission_counts[tag]:
                    emission_counts[tag][token] = 0
                emission_counts[tag][token] += 1
    
    #change the counts to probabilities using the MLE formula with the smoothing value of 0.01
    emission_probs = {}
    for state in hidden_states:
        emission_probs[state] = {}
        for token, count in emission_counts[state].items():
            emission_probs[state][token] = (count + 0.01)/(sum(emission_counts[state].values()) + 0.01 * (len(emission_counts[state]) + 1)) 
    
    with open("output_probs.txt", "w") as output:
        for state, token_dict in emission_probs.items():
            for token, prob in token_dict.items():
                output.write(f'{token} {state} {prob}\n')
    
output_probs('twitter_train.txt', 'twitter_tags.txt')
        
#(b)
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    #get the hidden states
    hidden_states = []
    hidden_states_index = {}
    with open(in_tags_filename, 'r') as tags: 
        for line in tags:
            hidden_states.append(line.strip())
            hidden_states_index[hidden_states[-1]] = len(hidden_states) - 1

    # get the output probabilities from the output_probs.txt file and store them in a dictionary for easy access
    output_probs = {}
    with open(in_output_probs_filename, 'r') as out_probs:
        for line in out_probs:
            token, tag, prob = line.split()
            if token in output_probs:
                output_probs[token][hidden_states_index[tag]] = float(prob)
            else:
                output_probs[token] = {hidden_states_index[tag]: float(prob)}

    # get the transition probabilities from the trans_probs.txt file and store them in a 2D list for easy access
    transition_probs = [[0] * len(hidden_states) for _ in range(len(hidden_states))]
    initial_probs = [0] * len(hidden_states)
    stop_probs = [0] * len(hidden_states)
    with open(in_trans_probs_filename, 'r') as trans_probs:
        for line in trans_probs:
            prev_tag, tag, prob = line.split()
            if prev_tag == "start":
                if tag == "stop":
                    continue
                initial_probs[hidden_states_index[tag]] = float(prob)
            else :
                if tag == "stop":
                    stop_probs[hidden_states_index[prev_tag]] = float(prob)
                else:   
                    transition_probs[hidden_states_index[prev_tag]][hidden_states_index[tag]] = float(prob)

    # define a nested function to calculate the viterbi algorithm
    def viterbi(sentence, hidden_states, transition_probs, output_probs, initial_probs, stop_probs):
        # initialise the viterbi table
        viterbi_table = [[]]
        backpointer = [[]]

        # getting emission probabilities 
        emission_probs = {}
        for tag in hidden_states:
            tag_num = hidden_states_index[tag]
            emission_probs[tag_num] = {}
            for token in sentence:
                if token in output_probs:
                    if tag_num in output_probs[token]:
                        emission_probs[tag_num][token] = output_probs[token][tag_num]
                    else:
                        # for unseen emission, assign a very small value
                        emission_probs[tag_num][token] = 0.000001
                else:
                    # for unseen token, assign a small value
                    emission_probs[tag_num][token] = 0.001

        # viterbi algorithm
        for i in range(0, len(hidden_states)):
            viterbi_table[0].append(initial_probs[i] * emission_probs[i][sentence[0]])
            backpointer[0].append(0)
        
        for t in range(1, len(sentence)):
            viterbi_table.append([])
            backpointer.append([])
            for s in range(0, len(hidden_states)):
                prob_values = []
                for s_prime in range(0, len(hidden_states)):
                    prob_values.append(viterbi_table[t - 1][s_prime] * transition_probs[s_prime][s] * emission_probs[s][sentence[t]])
                viterbi_table[t].append(max(prob_values))
                backpointer[t].append(prob_values.index(max(prob_values)))

        stop_prob_values = []
        for s in range(0, len(hidden_states)):
            stop_prob_values.append(viterbi_table[-1][s] * stop_probs[s])
        maxProb = max(stop_prob_values)
        finalBp = stop_prob_values.index(maxProb)

        #backtrack
        predicted_tags = [] 
        predicted_tags.append(hidden_states[finalBp])
        for t in range(len(sentence) - 1, 0, -1):
            predicted_tags.append(hidden_states[backpointer[t][finalBp]])
            finalBp = backpointer[t][finalBp]
        return predicted_tags[::-1]

    # read the test data and store them in a list of sentences
    tweet_sentences = [[]]
    with open(in_test_filename, 'r') as file:
        for line in file: 
            token = line.strip()
            if len(token) == 0:
                tweet_sentences.append([])
            else:
                tweet_sentences[-1].append(token)

    predictions = []
    for sentence in tweet_sentences:
        if len(sentence) == 0:
            continue
        # run the viterbi algorithm for each sentence
        predicted_tags = viterbi(sentence, hidden_states, transition_probs, output_probs, initial_probs, stop_probs)
        for tag in predicted_tags:
            predictions.append(f'{tag}\n')
        predictions.append('\n')
        
    with open(out_predictions_filename, 'w') as output:

       output.writelines(predictions)

# (c) Viterbi prediction accuracy:   1047/1378 = 0.7597968069666183

'''
Question 4

- 1st improvement: We've simplified the representation of user handles by collapsing all instances starts with @USER to "@USER". 
- 2nd improvement: We collapsed all token "http",to "http" as URLS can be unique and can overcomplicate prediction for U.
- 3rd improvement: Group all hashtags into one token "#" that if token startswith "#" return "#"
- 4th improvement:  Groups repeated characters in emoticons to a single character.      e.g. :)))) -> :), :(( -> :(
- 5th improvement: Groups repeated characters in punctuation to a single character.     e.g. !!!! -> !, ????? -> ?
- 6th improvement:  Identifies tokens that are mostly numeric. If more than half of the characters in the token are digits, returns "100", otherwise returns False.
'''

trans_probs('twitter_train.txt', 'twitter_tags.txt', 'trans_probs2.txt')

def URL_identifier(token):
        if token.startswith('http'):
            return 'http'
        else:
            return False
        

def USER_identifier(token):
    if token.startswith('@USER'):
        return '@USER'
    else:
        return False

def hashtag_identifier(token):
    if token[0] == "#":
        return '#'
    else:
        return False

def emoticon_shortener(token, tag):
    if tag == "E":
        cleaned_token = ''
        for i in range(len(token)):
            if i == 0 or token[i] != token[i-1]:
                cleaned_token += token[i]
        return cleaned_token
    else:  
        return False
    
def repeated_punctuation_shortener(token, tag):
    if tag == ",": #punctuation
        cleaned_token = ''
        for i in range(len(token)):
            if i == 0 or token[i] != token[i-1]:
                cleaned_token += token[i]
        return cleaned_token
    else:
        return False
    
def mostly_numeric(token):
    if sum(char.isdigit() for char in token) > len(token) / 2:
        return "100"
    else:
        return False

def output_probs2(in_train_filename, in_tags_filename):
    hidden_states = []
    with open(in_tags_filename, 'r') as tags:
        hidden_states = [line.strip() for line in tags]
    
    emission_counts = {}

    for state in hidden_states:
        emission_counts[state] = {}

    with open(in_train_filename, 'r') as train:
        for line in train:
            tokens = line.split()
            if len(tokens) >= 2:
                token = tokens[0]
                tag = tokens[1]
                if USER_identifier(token):
                    token = '@USER'
                if hashtag_identifier(token):
                    token = '#'
                if emoticon_shortener(token, tag):
                    token = emoticon_shortener(token, tag)
                if repeated_punctuation_shortener(token, tag):
                    token = repeated_punctuation_shortener(token, tag)
                if URL_identifier(token):
                    token = "URL"
                if mostly_numeric(token):
                    token = mostly_numeric(token)
                
                if token not in emission_counts[tag]:
                    emission_counts[tag][token] = 0
                emission_counts[tag][token] += 1
    
    #change the counts to probabilities
    emission_probs = {}
    for state in hidden_states:
        emission_probs[state] = {}
        for token, count in emission_counts[state].items():
            emission_probs[state][token] = (count + 0.01)/(sum(emission_counts[state].values()) + 0.01 * (len(emission_counts[state]) + 1)) 
    
    with open("output_probs2.txt", "w") as output:
        for state, token_dict in emission_probs.items():
            for token, prob in token_dict.items():
                output.write(f'{token} {state} {prob}\n')
    
output_probs2('twitter_train.txt', 'twitter_tags.txt')


def non_word_shortener(token):
    #if token does not contain any letters or numbers
    if not any(char.isalpha() or char.isdigit() for char in token):    
        cleaned_token = ''
        for i in range(len(token)):
            if i == 0 or token[i] != token[i-1]:
                cleaned_token += token[i]
        return cleaned_token
    else:
        return False

def tweet_preprocessor(tweet):
    tweet_sentences = [[]]
    with open(tweet, 'r') as file:
        for line in file:
            token = line.strip()
            if len(token) == 0:
                tweet_sentences.append([])
                continue
            if non_word_shortener(token):
                token = non_word_shortener(token)
            if URL_identifier(token):
                token = "URL"
            if USER_identifier(token):
                token = '@USER'
            if hashtag_identifier(token):
                token = '#'
            if mostly_numeric(token):
                token = mostly_numeric(token)
            tweet_sentences[-1].append(token)
    return tweet_sentences

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    #get the hidden states
    hidden_states = []
    with open(in_tags_filename, 'r') as tags: 
        for line in tags:
            hidden_states.append(line.strip())

    output_probs = {}
    with open(in_output_probs_filename, 'r') as out_probs:
        for line in out_probs:
            token, tag, prob = line.split()
            if token in output_probs:
                output_probs[token][hidden_states.index(tag)] = float(prob)
            else:
                output_probs[token] = {hidden_states.index(tag): float(prob)}

    transition_probs = {}
    initial_probs = {}
    stop_probs = {}
    with open(in_trans_probs_filename, 'r') as trans_probs:
        for line in trans_probs:
            prev_tag, tag, prob = line.split()
            if prev_tag == "start":
                if tag == "stop":
                    continue
                initial_probs[hidden_states.index(tag)] = float(prob)
            else :
                if tag == "stop":
                    stop_probs[hidden_states.index(prev_tag)] = float(prob)
                else:   
                    if hidden_states.index(prev_tag) in transition_probs:
                        transition_probs[hidden_states.index(prev_tag)][hidden_states.index(tag)] = float(prob)
                    else:
                        transition_probs[hidden_states.index(prev_tag)] = {hidden_states.index(tag): float(prob)}
    def viterbi(sentence, hidden_states, transition_probs, output_probs, initial_probs, stop_probs):
        #initialise the viterbi table
        viterbi_table = [[]]
        backpointer = [[]]

        #getting emission probabilities for this sentence
        emission_probs = {}
        for tag in hidden_states:
            tag_num = hidden_states.index(tag)
            emission_probs[tag_num] = {}
            for token in sentence:
                if token in output_probs:
                    emission_probs[tag_num][token] = output_probs[token][tag_num]
                else:
                    emission_probs[tag_num][token] = 0.01

        for i in range(0, len(hidden_states)):
            viterbi_table[0].append(initial_probs[i] * emission_probs[i][sentence[0]])
            backpointer[0].append(0)
        
        for t in range(1, len(sentence)):
            viterbi_table.append([])
            backpointer.append([])
            for s in range(0, len(hidden_states)):
                prob_values = []
                for s_prime in range(0, len(hidden_states)):
                    prob_values.append(viterbi_table[t - 1][s_prime] * transition_probs[s_prime][s] * emission_probs[s][sentence[t]])
                viterbi_table[t].append(max(prob_values))
                backpointer[t].append(prob_values.index(max(prob_values)))

        stop_prob_values = []
        for s in range(0, len(hidden_states) - 1):
            stop_prob_values.append(viterbi_table[-1][s] * stop_probs[s])
        maxProb = max(stop_prob_values)
        finalBp = stop_prob_values.index(maxProb)

        #backtrack
        predicted_tags = []
        predicted_tags.append(hidden_states[finalBp])
        for t in range(len(sentence) - 1, 0, -1):
            predicted_tags.insert(0, hidden_states[backpointer[t][finalBp]])
            finalBp = backpointer[t][finalBp]
        return predicted_tags
    
    tweet_sentences = tweet_preprocessor(in_test_filename)
    predictions = []
    for sentence in tweet_sentences:
        if len(sentence) == 0:
            continue
        predicted_tags = viterbi(sentence, hidden_states, transition_probs, output_probs, initial_probs, stop_probs)
        for tag in predicted_tags:
            predictions.append(f'{tag}\n')
        predictions.append('\n')
        
    with open(out_predictions_filename, 'w') as output:
        output.writelines(predictions)




def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)




def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '/Users/caophuong/Documents/Y2S2/BT3102/Project' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    

if __name__ == '__main__':
    run()
