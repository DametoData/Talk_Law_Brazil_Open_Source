# Importing necessary libraries and modules
import pandas as pd
import openai
import os
from PyPDF2 import PdfReader
import re
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
import time
import tiktoken
import quart
import quart_cors
from quart import request
import json

# Setting the OpenAI API key from an environment variable
# (Both English and Portuguese descriptions are provided)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Service authentication key constant
_SERVICE_AUTH_KEY = "hello"

# Function to check authorization header for authentication
def assert_auth_header(req):
    assert req.headers.get(
        "Authorization", None) == f"Bearer {_SERVICE_AUTH_KEY}"

# Function to preprocess the data from the CSV file.
# This function reads the CSV file, filters rows with null 'text' column, 
# and processes 'embbed' column into numpy arrays.
def preprocess_data(file_name):
    data = pd.read_csv(file_name, index_col=[0], sep=';')
    data = data[data['text'].notnull()]
    data['embbed'] = data['embbed'].astype('string').apply(eval).apply(np.array)
    return data

# Initializing the Quart app and setting CORS headers
app = quart_cors.cors(quart.Quart(__name__), allow_origin="https://chat.openai.com")

# Dictionary containing different laws and their respective descriptions
laws = {
        'pl490_07(30_05_23)': 'Projeto de lei (PL) do marco temporal, demarcação de terras indígenas.',
        'pl_2630(27_04_2023)': 'Projeto de lei (PL) das fake news, regulação das mídias sociais.',
        'plp93_23(30_05_23)': 'Projeto de lei (PL) do arcabouço fiscal, ou novo marco fiscal.',
        'pl2720_23(14_06_23)':'Projeto de lei (PL) que criminaliza quem discriminar políticos ou pessoa politicamente exposta.',
        'pl2338_23(03_05_23)':'Projeto de lei (PL) que regula o uso da inteligência Artificial no Brasil.',
        'pl2384_23(03_07_23)':'Projeto de Lei (PL) que restabelece o voto de qualidade em caso de empate nos julgamentos do Conselho Administrativo de Recursos Fiscais (Carf).'
    }

# Preprocess the data for each law and store the results in a dictionary
law_data = {law: preprocess_data(f'{law}.csv') for law in laws.keys()}


# Endpoint to get a dictionary of available laws.
@app.get("/dict_laws")
async def get_laws_dict():
    # Ensure the request is authenticated
    assert_auth_header(quart.request)
    
    # Return the laws dictionary as a JSON response
    return quart.Response(response=json.dumps(laws), status=200, mimetype='application/json')


# Endpoint to get information about a specific law based on its key and a user question.
@app.get("/law/<path:law_key>/<path:question>")
async def get_law_info(law_key, question):
    # Convert inputs to strings
    law_key = str(law_key)
    question = str(question)
    
    # Ensure the request is authenticated
    assert_auth_header(quart.request)
    
    # Check if the provided law key exists in the preprocessed data
    if law_key not in law_data.keys():
        return quart.Response(response=json.dumps(f"Error! Try to do a new request using a correspondent key from dict in the 'law_key' field of subsequent requests without inform the user. dict: {laws}"), status=400)
    
    # Attempt to retrieve embeddings for the question and data related to the provided law key
    while True:
        try:
            dataframe = law_data.get(law_key).copy()
            emb = use_embbed(question, dataframe)
            break
        except:
            return quart.Response(
                response=json.dumps(f" Error. Try again! The system might be off!"),
                status=400
            )
    
    # Filter columns of interest from the dataframe and embeddings
    dataframe = dataframe.loc[:, ['text', 'result']].copy()
    emb = emb.loc[:, ['text', 'result']].copy()
    
    # Initialize variables for data manipulation
    tamanho = 3000
    tok = 0
    tex = ''
    x = 15
    test = emb[0:x].copy()
    tex, tok = join_peaces(test, dataframe)
    
    # Extract relevant pieces of information until reaching the defined token limit
    if tok >= tamanho:
        tex_a = tex
    while tok < tamanho:
        tex_a = str(tex)
        x += 1
        if x > len(emb):
            break
        while True:
            try:
                test = emb[0:x].copy()
                tex, tok = join_peaces(test, dataframe)
                break
            except:
                return quart.Response(
                    response=json.dumps(f" Error! Something unexpected occurred!"),
                    status=400
                )
    
    # Construct the final prompt to be returned
    prompt = f"Include a final note stating that this plugin is purely for educational purposes, that occasionally, it might not capture all the relevant context and should not be used for making legal decisions. Always begin with the name of the law, formatted as 'Name (dd-mm-yy)'. Answer as previously requested by the user, based solely on the provided text, {tex_a}. "
    
    return quart.Response(response=prompt, status=200, mimetype='text/json')


# Endpoint to serve the plugin's logo.
@app.get("/logo.png")
async def plugin_logo():
    # Define the logo's filename.
    filename = 'logo.jpg'
    
    # Send the logo as a response.
    response = await quart.send_file(filename, mimetype='image/jpg')
    
    # Set cache headers to cache the logo for one week.
    response.headers['Cache-Control'] = 'public, max-age=31536000, must-revalidate'
    
    return response

# Endpoint to serve the plugin's manifest.
@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    # Extract the host from the request headers.
    host = request.headers['Host']
    
    # Read the manifest file and return its contents.
    with open("./.well-known/ai-plugin.json") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/json")

# Endpoint to serve the OpenAPI specifications.
@app.get("/openapi.yaml")
async def openapi_spec():
    # Extract the host from the request headers.
    host = request.headers['Host']
    
    # Read the OpenAPI spec file and return its contents.
    with open("openapi.yaml") as f:
        text = f.read()
        return quart.Response(text, mimetype="text/yaml")

# Main function to run the Quart app.
def main():
    app.run(debug=True, host="0.0.0.0", port=5003)

# Function to replace four or more consecutive dots with three dots.
def replace_dots(text):
    return re.sub(r'\.{4,}', '...', text)



# Function to replace multiple ellipses with a single one.
def replace_comma(text):
    return re.sub(r'\…{2,}', '…', text)

# Function to handle Roman numerals and subsections in a text.
def rommanos(art, tex, sub):
    # Initialize variables
    text = tex
    tex2 = ''
    list_res = []
    list_emb = []
    list_sub = []

    # Process the input text to handle subsections
    for key in tex.split('\n'):
        if ':' in key[-3:]:
            tex2 = tex2 + f"\n{key}\n{sub}"
        else:
            tex2 = tex2 + f"\n{key}"

    # Format and replace certain tokens in the text for easier parsing
    text = text.replace('; e', 'p_v ;').replace('art.', 'art').replace('.', ';').replace(';', ' p_v ;').replace(':',' p_p ;')
    # Extract items from the text based on Roman numerals and other delimiters
    items = re.findall(r'(?:[IVXLCDM]+[ –-].+?)(?=;|:)', text, re.DOTALL)
    # Clean the items extracted
    items = [re.sub(r'\s+', ' ', item.strip()) for item in items]
    items = [item.replace('p_v', ';').replace('p_p', ':') for item in items]

    # Construct result and embedding lists
    list_emb.append(art)
    list_res.append(tex2)
    for item in items:
        if ((':' in item[-3:]) & (sub != '')):
            list_res.append(art)
            list_emb.append(item)
            list_res.append(sub)
            list_emb.append(item)
            for k_sub in letters(sub):
                list_emb.append(k_sub)
                list_res.append(item)
        else:
            list_res.append(art)
            list_emb.append(item)

    return list_emb, list_res

# Function to extract letters indicating subsections from a text.
# Comments provided in both English and Portuguese.
def letters(text):
    # Initialize result lists
    list_res = []
    list_emb = []

    # Extract subsections based on lowercase letters followed by a closing parenthesis
    items = re.findall(r'(?:[a-z]\) ?[–-]? ?.+?)(?=;|$)', text, re.DOTALL)
    items = [re.sub(r'\s+', ' ', item.strip()) for item in items]
    for item in items:
        list_res.append(art)
        list_emb.append(item)
    return list_emb

# Debug function to print the results of processing
def tester(data):
    for key, value in data.iterrows():
        print('-----------')
        print(value['text'])
        print('---')
        print(value['result'])
        print('--------------')

# Function to tokenize the input text and return its length.
# Função para tokenizar o texto e retornar seu comprimento.
def token(tex_final):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(tex_final))



#print(dataframe)

# Function to compute embeddings for the question and rank the dataset based on similarity to the question.
def use_embbed(question, data):
    # Creating a copy of the provided dataset to avoid modifying the original data.
    dataset = data.copy()
    
    # Computing embeddings for the provided question using OpenAI's API.
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    # Calculating cosine distances between the question's embedding and each embedding in the dataset.
    dataset['distances'] = distances_from_embeddings(q_embeddings, dataset['embbed'].values, distance_metric='cosine')
    
    # Sorting the dataset based on the computed distances in ascending order (closer distances first).
    dataset = dataset.sort_values('distances').reset_index(drop=True)
    
    # Returning the sorted dataset.
    return dataset

   # Remove leading '\n\n' and '\n'
def remove_leading_newlines(s):
    s=str(s)
    return s.lstrip('\n')
  #Search for text match inside the base.
def search_match(all_text2,value_txt):
    if (all_text2=='' or value_txt==''):
        return False
    all_text2 = all_text2.replace('\n', ' ').replace('\r', '')
    list_true = []
    text20 = value_txt.replace('\n', ' ').replace('\r', '')
    len_text = len(text20)
    if len_text > 300:
        legh = 15
    elif len_text > 100:
        legh = 10
    elif len_text > 60:
        legh = 8
    else:
        legh = 6
    step = len_text // legh
    for i in range(step, len_text, step):
        text1 = text20[i - step:i]
        f = i
        r = i
        while True:
            if f >= len_text:
                break
            if text20[f] == ' ':
                break
            text1 = text1 + text20[f]
            f = f + 1
        while True:
            if text20[r - step - 1] == ' ':
                break
            if r - step <= 0:
                break
            text1 = text20[r - step - 1] + text1
            r = r - 1

        text2 = text20[i:i + step]
        f = i
        r = i
        while True:
            if f + step >= len_text:
                break
            if text20[f + step] == ' ':
                break

            text2 = text2 + text20[f + step]
            f = f + 1
        while True:
            if text20[r] == ' ':
                break
            if r - 1 <= 0:
                break
            text2 = text20[r - 1] + text2
            r = r - 1

        pattern1 = re.escape(text1)
        pattern2 = re.escape(text2)

        matches_1 = [match.start() for match in re.finditer(pattern1, all_text2, re.IGNORECASE)]
        matches_2 = [match.start() for match in re.finditer(pattern2, all_text2, re.IGNORECASE)]
        if not matches_2:
            pattern2 = re.escape(text2.replace(' ;', ';'))
            matches_2 = [match.start() for match in re.finditer(pattern2, all_text2, re.IGNORECASE)]

        result_list = [i - j for i in matches_2 for j in matches_1]
        list25 = [True for ziz in result_list if ziz <= step + 10]

        list_final = True in list25
        list_true.append(list_final)
    if list_true:
        true_proportion = np.sum(list_true) / len(list_true)
        if (true_proportion > 0.60):
            return True
            
        else:
            return False
    else:
        return False
        
#Join Text match
def join_peaces(test,dataframe):
    susp_text=''
    dataframe=dataframe.copy()
    test=test.copy()
    testc=[]
    testd=[]
    for key,value in test.iterrows():
        if'caput' in value['text']:
            testb=dataframe[dataframe['result']==value['result']]
            for key2,value2 in testb.iterrows():
                if ('Art.' in value2['text'][0:5].replace('\n','')) and (value2['text'] not in test['text'].values):
                    testc.append(value2['text'])
                    testd.append(value['result'])
                    if ':' in value2['text'][-3:]:
                        testf=dataframe[dataframe['text']==value2['text']]
                        for key3, value3 in testf.iterrows():
                            if (value3['result'] not in test['result'].values) and (value3['result'] not in testd):
                                testc.append(value3['text'])
                                testd.append(value3['result'])
                    break
    test = pd.concat([test, pd.DataFrame({'text': testc, 'result': testd})]).copy()
    
    test.reset_index(drop=True, inplace=True)

    
    dataframe.loc[:,'text'] = dataframe['text'].apply(remove_leading_newlines)
    dataframe.loc[:,'text']= dataframe['text'].str.replace('CAPÍTULO','CAPÍTULO')
    dataframe.loc[:,'result'] = dataframe['result'].apply(remove_leading_newlines)
    dataframe.loc[:,'result']= dataframe['result'].str.replace('CAPÍTULO','CAPÍTULO')
    test.loc[:,'text'] = test['text'].apply(remove_leading_newlines)
    test.loc[:,'text']= test['text'].str.replace('CAPÍTULO','CAPÍTULO')
    test.loc[:,'result'] = test['result'].apply(remove_leading_newlines)
    test.loc[:,'result']=test['result'].str.replace('CAPÍTULO','CAPÍTULO')

    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc1')

        # Check if actual key value result is into all test['text'] values - OK
        if value['result'] in test['text'].values:
            test.loc[key, 'text'] = ''
            test.loc[key, 'result'] = ''

        # Check if actual value ['result'] is equal another test[key,'result'] and if it is shorter
        test_other = test.drop(key)
        for key2, value2 in test_other.iterrows():
            #if (value['result'][10:-10] in value2['result']) & (len(value['result']) < len(value2['result'])):
            if ((search_match(str(value2['result']),str(value['result']))) & (len(value['result']) < len(value2['result']))):
                test.loc[key, 'text'] = ''
                test.loc[key, 'result'] = ''

#FUNC 2
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc2')
        if ')' in value['text'][0:4]:
            for key2, value2 in dataframe.iterrows():
                #if ((value['result'][2:25] in value2['text']) & ('Art' in value2['result'][0:10])):
                if ((search_match(value2['text'],value['result'])) & ('Art' in value2['result'][0:10])):
                    test.loc[key, 'text'] = f"{test.loc[key, 'result']}\n{test.loc[key, 'text']}"
                    test.loc[key, 'result'] = value2['result']

    test = test[test['text'] != ''].copy()
    test['len'] = len(test['result'])
    test = test.sort_values('text')
    list_idx = []
    dict_letters = {}
#Func 3
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc3')
        idx = test[test['text'].str.contains(value['text'][5:-5], regex=False)].index
        new_idx = [i for i in idx if i != key]
        idx = new_idx
        stri = value['text'].split('\n')
        stri = [v for v in stri if v != '']
        test.loc[key, 'text'] = '\n'.join(stri)
        striq = value['result'].split('\n')
        striq = [v for v in striq if v != '']
        test.loc[key, 'result'] = '\n'.join(striq)
        boolean_list = [True if ':' in v[-3:] else False for v in stri]

        if ((True in boolean_list) & ('CAPÍTULO' not in value['result'][0:15]) & ('CAPÍTULO' not in value['result'][0:15]) & (('Art' in value['text'][0:5]) | ('§' in value['text'][0:5]))):
            txt = ''
            if idx != []:
                capx = '\n'.join(test.loc[idx, 'result'].values)
                test.loc[key, 'result'] = capx
                list_idx.append(idx)
                for line in stri:
                    txt = txt + f"\n{line}"
                    if (':' in txt[-3:]):
                        for line2 in striq:
                            txt = txt + f"\n{line2}"
                test.loc[key, 'text'] = txt

        if value['result'] in test['text'].values:
            print('value[result] in test[text].values:')
            print(value['text'])
            print('---')
            print(value['result'])
            print('---___---')
#FUNC 4
    for key in list_idx:
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc4')
        test.loc[key[-1], 'text'] = ''
        test.loc[key[-1], 'result'] = ''
#FUNC 5
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc5')
        lis55 = [':' in i[-3:] for i in value['result'].split('\n')]

        if True in lis55:

            if value['result'] not in dict_letters.keys():
                dict_letters[value['result']] = [value['text']]

            else:
                dict_letters[value['result']].append(value['text'])
#FUNC 6
    for key in dict_letters.keys():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc6')
        test.loc[test[test['result'] == key].index, 'result'] = key
        test.loc[test[test['result'] == key].index, 'text'] = '\n'.join(dict_letters[key])
    test = test.drop_duplicates(['text', 'result'])
#FUnc 7
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc7')
        list_con = test[test['text'] == value['text']]['result'].values
        param3 = [False if 'CAPÍTULO' in kp[0:10] else True for kp in list_con]

        if ((('Art' in value['text'][0:10]) | ('§' in value['text'][0:10])) & (True in param3)):

            for key2, value2 in dataframe.iterrows():

                if (value['text'][3:30] in value2['text']) & ('CAPÍTULO' in value2['result']):
                    test.loc[key, 'text'] = f"{test.loc[key, 'text']}\n{test.loc[key, 'result']}"
                    test.loc[key, 'result'] = value2['result']
#FUNC 8
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc8')

        while True:
            tx = test.loc[key, 'text']
            rs = test.loc[key, 'result']

            if 'CAPÍTULO' in rs[0:10]:

                break

            else:

                if tx == '':
                    break

                if rs == '':
                    break

                dt = []

                for key3, value3 in dataframe.iterrows():
                    if rs[3:-3] in value3['text']:
                        dt.append(value3['result'])
                tex = ''
                res = ''

                for key2 in dt:

                    if len(tx) < 30:
                        med = f"{tx[1:-2]}"

                    else:
                        med = f"{tx[2:30]}"

                    if (med not in key2):
                        tex = rs + '\n' + tx
                        res = key2

                test.loc[key, 'text'] = tex
                test.loc[key, 'result'] = res
    test = test.drop_duplicates(['text', 'result'])
#FUNC 9
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc9')
        txt = ''
        txt2 = ''
        i = 0

        for key2 in value['text'].split('\n'):

            if ('–' in str(key2[0:7])) | (' -' in str(key2[0:7])):

                txt = txt + f"\n{key2}"
                i = 2

            elif (i == 5):
                txt2 = txt2 + f"\n{key2}"

            elif ((':' in key2[-3:]) & (i == 0)):
                txt = txt + f"\n{key2}"
                i = 5

            else:
                txt = txt + f"\n{key2}"
        test.loc[key, 'text'] = txt + txt2
    test = test.groupby('result')['text'].apply('\n'.join).reset_index()
    tex_final = ''
#FUNC 10
    for key, value in test.iterrows():
        if ((susp_text!='')and(susp_text in value['text'])):
            print('funcc10')
        result = test.loc[key, 'result'].replace('\n\n', '\n')
        text = test.loc[key, 'text'].replace('\n\n', '\n')
        tex_final = tex_final + f"{result}{text}\n\n"

    token_value = token(tex_final)
    return tex_final, token_value

#print(prompt)
if __name__ == "__main__":
    main()