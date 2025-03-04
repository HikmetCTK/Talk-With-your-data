import pandas as pd 
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def find_file_type(df_path):
    # Get path of file
    _, extension = os.path.splitext(df_path)
    
    try:
        # Check extension and read file 
        if extension.lower() == '.csv':
            df = pd.read_csv(df_path)
            return df, "CSV dosyası başarıyla okundu."
        elif extension.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(df_path)
            return df, "Excel dosyası başarıyla okundu."
        else:
            raise ValueError("Desteklenmeyen dosya türü.")
    except Exception as e:
        return None, f"Hata: {str(e)}"


def convert_to_pandas(user_query, df):# Text2Pandas code agent
    """
    args:
    user_query(str):user query
    df:dataframe
    description: This Function is used for converting user question to pandas code
    Returns:True,pandas_query(str) or  False,pandas_query(warning message)
    """
    
    columns = df.columns.tolist()  # Get list of columns name of dataset
    
    system_prompt = """
    Sen Kullanıcının isteğini uygun Pandas formatına dönüştüren yardımsever bir asistansın.Sadece veritabanıyla ilgili gelen  sorulara cevap ver.Veri tabanında değişiklik yapmak istersen buna asla izin verme  . Pandas kodun sana verilen tablo ismine ve kolonlarına uygun olmalıdır.
    **aşağıda ki izin verilmeyen komutları asla kullanma
    izin_verilmeyen_islemler =loc,iloc,replace,fillna,drop,apply

    Tablos ismin:df
    Kolon isimleri: {df_columns}
    Yanıtın json formatında olmalı
    *Json formatı:*
    {{
    "query": "Ali Yılmaz hangi departmanda çalışıyor",
    "pandas_query": "df[df['Kişi'] == 'Ali Yılmaz']['Departman'].unique()[0]",
    "confidence":0.85,
    "query":"Bugün nasılsın",
    "pandas_query":" Garip Garip alakasız sorular sorma Dostum Verisetinle ilgili bir şey sor.🤨🤨🤨.",
    "confidence":0.2
    }}
    """
    
    # Değişkenleri yerleştirmek için promptu formatla
    formatted_system_prompt = system_prompt.format(df_columns=columns)
    response=client.models.generate_content(model="gemini-2.0-flash",contents=user_query,
                                        config=types.GenerateContentConfig(
                                            temperature=0.1,
                                            top_k=64,
                                            top_p=0.96,
                                            response_mime_type="application/json",
                                            system_instruction=formatted_system_prompt))
    try:
        json_response=json.loads(response.text)

        pandas_query=json_response["pandas_query"] # pandas code

        confidence=json_response["confidence"]  # Confidence score

        if confidence>0.7:  
            return True,pandas_query
        else:
            
            return False,pandas_query
    except Exception as e:
        return "",str(e)
    
def convert_to_visual_code(user_query, df):#  Text to Visualizaton agent
    """
    args:
    user_query:str(user query)
    df:dataframe
    description: This function is used for converting user query to visualization code or false based on query.
    Returns:visual_code(str):Visualization code or False
    """
    
    summary=pd.DataFrame({"Data types":df.dtypes,
                      "Unique values":df.nunique()}) # Data column infos and unique value count 
    
    system_prompt = """
    * Sen Kullanıcının isteğini uygun görselleştirme koduna  dönüştüren yardımsever bir  görselleştirici asistansın.
    * Sadece veritabanıyla ilgili gelen  sorulara cevap ver.Veri tabanında değişiklik yapmak istersen buna asla izin verme. 
    * Görselleştirme kodların  aşağıda belirtilen kolon bilgileri ile uyumlu olmalıdır. 
    *Görselleştirme dışında ki isteklere sns.pairplot(df) kodunu yaz*


    Kullanacağın kütüphaneler:matplotlib.pyplot(plt),seaborn(sns)
    Veritabanı ismin:df
    Kolon bilgileri: {summary}
    Yanıtın json formatında olmalı
    *Json formatı:*
    {{
    "query": "departman bilgilerini pie plot ta göster",
    "visual_code": "plt.pie(df, labels=df["departman"].index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'orange', 'lightgreen', 'red'])",
    "confidence":0.85,
    "query":"bugün nasılsın"
    "visual_code": "sns.pairplot(df)",
    "confidence":0.85
    }}

    """
    
    # Formatted prompt with column info
    formatted_system_prompt = system_prompt.format(summary=summary)
 
    formatted_system_prompt = system_prompt.format(summary=summary)
    response=client.models.generate_content(model="gemini-2.0-flash",contents=[user_query],
                                        config=types.GenerateContentConfig(
                                            temperature=0.1,
                                            top_k=64,
                                            top_p=0.96,
                                            response_mime_type="application/json",
                                            system_instruction=formatted_system_prompt))
    try:
        json_response=json.loads(response.text)
        
        visual_code=json_response["visual_code"] # pandas code

        confidence=json_response["confidence"]  # Confidence score
        if confidence>0.7:  
            return visual_code
        else:
            return False
    except Exception as e:
        return str(e)



def safe_exec(code, local_vars):
    """ 
    args: 
    Code(str):pandas code
    
    Description: This function is used to prevent dangerous codes like (os,sys,subprocess) and return  result variable
    
    returns:
            local_vars["result"]:Result variable which includes output of pandas code
    """
    code="result="+code # Output is in result variable
    safe_builtins = { #permitted commands
        'len': len,
        'max': max,
        'min': min,
        'range': range,

    }
    try:
        exec(code, {"__builtins__": safe_builtins}, local_vars) # Libraries like os,sys,subprocess  are not permitted to run
    except Exception as e:
        return f"Hata: {str(e)}"
    return local_vars["result"]




def readable_answer_agent(user_query,response):
    """ 
    args:
        
        user_query(str):User question
        response(str):Output from safe_exec function
    description:Returns question's answer by taken from user in good format .
    returns:
        response.text(str):LLM response
    """
    system_prompt = """Sen, kullanıcının sorusu ve gelen cevabı dikkate alarak güzel açıklayıcı formatta dost canlısı cevap veren bir asistansın.Cümlelerin sonuna emoji koymayı unutma

    örnek senaryo:
    kullanıcı sorusu:en çok maaşı alan eleman kim
    cevap:Ahmet Çelik
    açıklayıcı cevap:En çok maaşı alan eleman Ahmet Çelik'tir💸

    """


    response = client.models.generate_content(model="gemini-1.5-flash",
                                              contents=[f"kullanıcı sorusu:{user_query},cevap:{response}"],
                                              config=types.GenerateContentConfig(temperature=0.7, # Cretaful Responses
                                                                                 system_instruction=system_prompt)
                                              )
    return response.text


def process_file_and_query(file, user_query):
    """ 
    args:
        file(csv,xls,xlsx):File which is selected by user
        user_query(str):User question
        
    Description:This function is used to combine all func into one function

    Returns:
        Final_answer(str):Response
        Pandas_code(str):If user ask unrelated question LLM returns warning message
            
    """
    df, message = find_file_type(file)
    
    if df is None:
        return message, None
    elif user_query=="":
        return "Bari bir şey sorsaydında basıverseydin",None
    else:
        state,pandas_code = convert_to_pandas(user_query=user_query, df=df)
        
        if state:   #   if  convert_to_pandas function  returned pandas code           
            local_vars = {"df": df}  # Only df variable can be modified
            output = safe_exec(pandas_code, local_vars)
            final_answer = readable_answer_agent(output, user_query)
            return final_answer
        else:               #if  convert_to_pandas function  returned False 
            return pandas_code #LLM warning response which is assigned in prompt


def exec_visual_code(query, file):
    """ 
    args:
        query(str):User query
        file(file):Dataset file
    description:Visualize the  user's question.
    returns:
        plt.gcf(Plot image):Plot image of executed visualization code
    """
    if query is None:
        return "Enter your query"
    
    df,msg= find_file_type(file)


    # Clear previous plots
    plt.clf()  # Clears current figure
    plt.close()  # Closes previous figures to prevent memory issues

    code = convert_to_visual_code(query, df)
    if code:
        exec(code)  # Execute the generated visualization code
    else:
        return ""
    
    return plt.gcf()  # Return the new figure

#Created this  Function for Gradio
def gradio_for_analysis(file, user_query):
    final_answer = process_file_and_query(file, user_query)

    return  final_answer

# Create the Gradio app
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column(scale=1):
                gr.Markdown("# Text to Code Data Agents🕵🏻🤖🎨")
                file_input=gr.File(label="Upload CSV Or Excel File ")

                query_input=gr.Textbox(label="Enter your query for Data Analysis")
                analyze_button=gr.Button("Get Answer")

                vis_query_input=gr.Textbox(label="Enter Your Query For Data Visualization")
                visualize_button=gr.Button("Visualize It📊📈📉")

    


        with gr.Column(scale=2):
            
                answer_output=gr.Textbox(label="Answer")
                vis_output=gr.Plot(label="Visualization")

    analyze_button.click(fn=gradio_for_analysis,inputs=[file_input,query_input],outputs=[answer_output])
    visualize_button.click(fn=exec_visual_code,inputs=[vis_query_input,file_input],outputs=[vis_output])



# Launch the Gradio app
iface.launch(share=False) 





