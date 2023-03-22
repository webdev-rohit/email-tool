from flask import Flask, request, render_template, redirect
import openai
import tensorflow as tf
import tensorflow_text as text
import json
tf.get_logger().setLevel('ERROR')
app = Flask(__name__)
import config

openai.api_key = config.api_key
model =  tf.saved_model.load('C:\Learnings\BERT_Email_Spam_Classification\FinalProjectFolder')

def summarize():
    request_body = request.form['emailbody']
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Summarize this email:\n\n"+str(request_body),
    temperature=1.0,
    max_tokens=50,
    top_p=0.5,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )   
    finalResponse = response['choices'][0]['text']
    print('final response: ',finalResponse)
    print('type of final response: ',type(finalResponse))
    return finalResponse

def predict():
    emailBody = [] 
    # request_body = request.get_data()
    request_body = request.form['emailbody']
    emailBody.append(request_body)
    print("Email body: ",emailBody)
    a = tf.sigmoid(model(tf.constant(emailBody)))
    print("value of a: ", a)
    if a>=0.68: 
        result = '{"result":"spam"}'
        jsonObj = json.loads(result)
        return jsonObj
    else:
        result = '{"result":"not spam"}'
        jsonObj = json.loads(result)
        return jsonObj
    
def getintents():
    request_body = request.form['emailbody']
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt = "Provide a JSON output determining the intents and entities from the following text::\nI'm writing to ask for annual leave in advance of my entitlements. I'd like to take my leave between the following dates:23rd Feb, 24th Feb. I'll be away for 2 weeks, which is in accordance with the company's annual leave policy.\n{\n  \"intents\": [\n\t\"RequestLeave\"\n  ],\n  \"entities\": [\n\t\t{\n\t\t  \"entity\": \"leave_type\",\n\t\t  \"value\": \"annual leave\"\n\t\t},\n\t\t{\n\t\t  \"entity\": \"start_date\",\n\t\t  \"value\": \"23rd Feb\"\n\t\t},\n\t\t{\n\t\t  \"entity\": \"end_date\",\n\t\t  \"value\": \"24th Feb\"\n\t\t},\n\t\t{\n\t\t  \"entity\": \"duration\",\n\t\t  \"value\": \"2 weeks\"\n\t\t}\n  ]\n}\n##\nDear Employee, \nEnclosed is your Relieving Letter. ID card Submission - Pls confirm on ID Card submission status.Was it sent through company organized pick-up service or submitted directly to the LTI office?  In case we do not receive any revert on the ID card submission status within 2 days from the receipt of this mail, we will go ahead with the recovery and close the HR clearance \n{\n  \"intents\": [\n    \"Relieving Letter\",\n    \"ID Card Submission\",\n    \"HR Clearance\"\n  ],\n  \"entities\": [\n\t  {\n\t\t\"entity\": \"service_type\",\n\t\t\"value\": \"Company Organized Pick-up Service\",\n\t  },\n\t  {\n\t\t\"entity\": \"company\",\n\t\t\"value\": \"LTI Office\",\n\t  {\n\t  \"entity\": \"duration\",\n\t  \"value\": \"2 days\"\n\t  }\n  ]\n}\n##\nhello rohit\n{\n  \"intents\": [\n   \"greeting\"\n  ],\n  \"entities\": [\n    {\n      \"entity\": \"person\",\n      \"value\": \"rohit\"\n    }\n  ]\n}\n##\nhello\n{\n  \"intents\": [\n      \"greeting\"\n  ],\n  \"entities\": []\n}\n##"+str(request_body),
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    finalResponse = response['choices'][0]['text']
    print('final response: ',finalResponse)
    print('type of final response: ',type(finalResponse))
    return finalResponse
           

@app.route("/", methods=['POST', 'GET'])
def index():
  if request.method == 'GET':
    return render_template('index.html')
  elif request.method == 'POST':
    sumresult  = summarize()
    print('summarizer result: ', sumresult)
    classifierjsonObj = predict()
    classresult = classifierjsonObj['result']
    print('classifier result: ', classresult)
    intentsjsonStr = getintents()
    print('intents json Str: ',intentsjsonStr)
    print('type of intents json Str: ',type(intentsjsonStr))
    try:
        intentsjsonObj = json.loads(intentsjsonStr)
        print(type(intentsjsonObj))
        intents = intentsjsonObj['intents']
        print(intents)
        entities = intentsjsonObj['entities']
        print(entities)
        return render_template('index.html', sumresult = sumresult, classresult = classresult, intents = intents, entities = entities)
    except:
        return 'Something went wrong! There seems to be an issue with the object returned by OpenAI. Kindly, try posting another email body or try after some time.'
    
if __name__=="__main__":
   app.run(debug=True)
