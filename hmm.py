# Implement the four functions below

#Question 2.1a

def MLE_predict(training_file):
   #create two dictionary to store the count of tags and the count of pair tags and words
   # read the training_file and split the words and tags 
    # for each word and tag pair, increment the count of the tag and the pair tag and word
    # tag_count = {tag: count(tag)}
    #pair_count = {token: {tag: count(token, tag)}}
    tag_count = {}
    pair_count = {}

    with open(training_file, 'r') as file:
        for line in file:
            line_elements = line.strip()
            if len(line_elements) >= 2:
                token, tag = line_elements.split()
                if tag in tag_count:
                    tag_count[tag] += 1
                else:
                    tag_count[tag] = 1
                if token in pair_count:
                    if tag in pair_count[token]:
                        pair_count[token][tag] += 1
                    else:
                        pair_count[token][tag] = 1
                else:
                    pair_count[token] = {tag: 1}
    
    #Dictionary of dictionary to store the output probabilties
    #key: token. value: {tag: P(token|tag)=count(token, tag)/count(tag)}
    #smoothing value: 0.01 
    result = {}
    for token in pair_count:
        result[token] = {}
        for tag in pair_count[token]:
            result[token][tag] = (pair_count[token][tag] + 0.1) / (tag_count[tag] + 0.1 * (len(pair_count) + 1))
    
    with open('naive_output_probs.txt', 'w') as outfile:
        for token in result:
            for tag in result[token]:
                outfile.write(f'{token} {tag} {result[token][tag]}\n')
MLE_predict('twitter_train.txt')

#Question 2.2b
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    output_probs = {}
    with open(in_output_probs_filename, 'r') as file:
        for line in file:
            token, tag, prob = line.strip().split()
            if token not in output_probs:
                output_probs[token] = {}
            output_probs[token][tag] = float(prob)
    print(output_probs)

    tweets = []
    with open(in_test_filename, 'r') as file:
        for line in file: 
            line = line.strip('\n')
            if line:
                tweets.append(line)

    predicted_tags = []
    for token in tweets:
        if token in output_probs:
            predicted_tag = max(output_probs[token], key=output_probs[token].get)
        else:
            if token.startswith('@'):
                predicted_tag = '@'
            else:
                #thinking about the case that if token is Summer, but in output_probs only only has summer
                predicted_tag = 'N' #default tag is Noun 
        predicted_tags.append(predicted_tag)
    
    with open(out_prediction_filename, 'w', encoding="utf-8") as outfile:
        for tag in predicted_tags:
            outfile.write(f'{tag}\n')

# Question 2.1.c
# Naive prediction accuracy:     1063/1378 = 0.7714078374455733


'''
Question 2.2.a
By Bayes' rule: P(y = j|x = w) = P(x = w|y = j)P(y = j) / P(x = w)
When comparing the probability of each tag j for each word, P(w) remains the same for every tag, hence we can ignore it.
MLE: P(j) = count(y = j) / count(y - total tags) - count số lượng j xuất hiện trong training data chia cho số lượng tất cả các tag
P(w|j) takes from naive_output_probs.txt
'''

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
            if token.startswith('@'):
                predicted_tag = '@'
            else:
                predicted_tag = 'N'
        predicted_tags2.append(predicted_tag)

    with open(out_prediction_filename, 'w', encoding="utf-8") as outfile:
        for tag in predicted_tags2:
            outfile.write(f'{tag}\n')
# Question 2.2.b
# Naive prediction2 accuracy:    1075/1378 = 0.7801161103047896


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass




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

    # trans_probs_filename =  f'{ddir}/trans_probs.txt'
    # output_probs_filename = f'{ddir}/output_probs.txt'

    # in_tags_filename = f'{ddir}/twitter_tags.txt'
    # viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    # viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
    #                 viterbi_predictions_filename)
    # correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    # print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    # trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    # output_probs_filename2 = f'{ddir}/output_probs2.txt'

    # viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    # viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
    #                  viterbi_predictions_filename2)
    # correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    # print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()
