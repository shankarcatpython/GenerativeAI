import openai
import config


def askme_questions_summarize(prompt):
    final_prompt = "Summarize the following paragraph" + prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are very good on summarizing"},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message['content']

def askme_questions_suggestion(prompt):
    final_prompt_sugg = "Suggest some thing from following paragraph" + prompt
    response_sugg = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are very good at suggesting things"},
            {"role": "user", "content": final_prompt_sugg}
        ]
    )

    return response_sugg.choices[0].message['content']


