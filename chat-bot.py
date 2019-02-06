#importing important libraries
import numpy as np
import tensorflow as tf
import re
import time

#Loading the dataset's conversation and lines text
lines=open('movie_lines.txt', encoding='utf-8',errors= 'ignore').read().split('\n')
conversations=open('movie_conversations.txt', encoding='utf-8',errors= 'ignore').read().split('\n')

id2line = {}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
conversation_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(_conversation.split(","))
#creating questions and answers
    
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
#creating a function for cleaning the text using it as questions and answers text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am", text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"[-()\+=?/,.:;<>{}@#~]","",text)
    return text

#creating cleaned questions list                
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
#creating cleaned answers list  
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))            
#creating a dictionary that maps word to number of occurences in both cleaned questions and answers
word2count={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
#creating a dictionary for questions and answers that will map it to a unique integer
threshold=20
questionswords2int={}
word_number=0
for word, count in word2count.items():
    if count>=threshold:
        questionswords2int[word]=word_number
        word_number+=1
answerswords2int={}
word_number=0
for word, count in word2count.items():
    if count>=threshold:
        answerswords2int[word]=word_number
        word_number+=1

tokens=['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:    
    questionswords2int[token]=len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token]=len(answerswords2int) + 1
#creating an inverse dictionary for answerswords2int
answersints2word={w_i: w for w, w_i in answerswords2int.items()}
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
questions_into_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
#creating a list of questions and answers based upon length of questions
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])

def model_inputs():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob

def preprocess_targets(targets,word2int,batch_size):
    left_side=tf.fill([batch_size,1],word2int['<SOS>'])
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1])
    preprocessed_targets=tf.concat([left_side,right_side],axis=1)
    return preprocessed_targets

def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,sequence_length=sequence_length,inputs=rnn_inputs,dtype=tf.float32)
    return encoder_state

def decoder_training_set(decoder_cell,encoder_state,decoder_embedded_inputs,sequence_length,decoding_scope,keep_prob,output_function,batch_size):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output_size)
    training_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name='attn_dec_train')
    decoder_output,decoder_final_state,decoder_context_final_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                          training_decoder_function,
                                                                                                          decoder_embedded_input,
                                                                                                          sequence_length,
                                                                                                          scope=decoding_scope)
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

def decoder_test_set(decoder_cell,encoder_state,decoder_embeddings_matrix,sos_id,eos_id,maximum_length,num_words,sequence_length,decoding_scope,keep_prob,output_function,batch_size):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                               attention_option='bahdanau',
                                                                                                                               num_units=decoder_cell.output_size)
    test_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_inference(encoder_state[0],
                                                                            output_function,
                                                                            sos_id,
                                                                            eos_id,
                                                                            maximum_length,
                                                                            num_words,
                                                                            decoder_embeddings_matrix,
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name='attn_dec_inf')
    test_predictions,decoder_final_state,decoder_context_final_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            test_decoder_function,
                                                                                                            decoder_embeddings_matrix,
                                                                                                            scope=decoding_scope,
                                                                                                            )
    return test_predictions

def decoder_rnn(encoder_state,decoder_embedded_input,decoder_embeddings_matrix,num_words,sequence_length,rnn_size,num_layers,keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell=tf.nn.MultiRNNCell([lstm_dropout]*num_layers)
        weights=tf.truncated_normal_initializer(stddev=0.1)
        biases=tf.zeros_initializer()
        output_function=lambda x: tf.contrib.layers.fully_connected(x,
                                                                    num_words,
                                                                    None,
                                                                    scope=decoding_scope,
                                                                    weights_initializers=weights,
                                                                    biases_initializer=biases)
        training_predictions=decoder_training_set(encoder_state,
                                                  decoder_cell,
                                                  decoder_embedded_inputs,
                                                  sequence_length,
                                                  decoding_scope,
                                                  output_function,
                                                  keep_prob,
                                                  batch_size)
        decoding_scope.reuse_variables()
        test_predictions=decoder_test_set(decoder_cell,
                                          encoder_state,
                                          decoder_embeddings_matrix,
                                          words2int['<SOS>'],words2int['<EOS>'],
                                          maximum_length,
                                          num_words,
                                          sequence_length,
                                          decoding_scope,
                                          rnn_size,
                                          num_layers,
                                          keep_prob,
                                          batch_size)
    return training_predictions, test_predictions

def seq2seq(inputs,targets,keep_prob,batch_size,sequence_length,answers_num_words,questions_num_words,encoder_embedding_size,decoder_embedding_size,rnn_size,num_layers,questionswords2int):
    encoder_embedded_input=tf.contrib.layers.embed_sequence(inputs,
                                                            answers_num_words + 1,
                                                            encoder_embedding_size,
                                                            initializer=tf.random_uniform_initializer(0,1))
    
    encoder_state=encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob,sequence_length)
    preprocessed_targets=preprocessed_targets(targets,questionswords2int,batch_size)
    
    